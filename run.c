/*
Inference for Llama 3.x Transformer model in pure C.
Use legacy export to produce compatible .bin files.
*/

#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <limits.h>
#if defined _WIN32
#include "win.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif
// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
  int dim;        // transformer dimension
  int hidden_dim; // for ffn layers
  int n_layers;   // number of layers
  int n_heads;    // number of query heads
  int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
  int vocab_size; // vocabulary size, usually 4096 (byte-level)
  int seq_len;    // max sequence length
} Config;

typedef struct {
  // token embedding table
  float *token_embedding_table; // (vocab_size, dim)
  // weights for rmsnorms
  float *rms_att_weight; // (layer, dim) rmsnorm weights
  float *rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  float *wq; // (layer, dim, n_heads * head_size)
  float *wk; // (layer, dim, n_kv_heads * head_size)
  float *wv; // (layer, dim, n_kv_heads * head_size)
  float *wo; // (layer, n_heads * head_size, dim)
  // weights for RoPE
  float *freqs_cos;   // (seq_len, n_heads)
  float *freqs_sin;   // (seq_len, n_heads)
  // weights for ffn
  float *w1; // (layer, hidden_dim, dim)
  float *w2; // (layer, dim, hidden_dim)
  float *w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  float *rms_final_weight; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  float *wcls;
} TransformerWeights;

typedef struct {
  // current wave of activations
  float *x;      // activation at current time stamp (dim,)
  float *xb;     // same, but inside a residual branch (dim,)
  float *xb2;    // an additional buffer just for convenience (dim,)
  float *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
  float *hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
  float *q;      // query (dim,)
  float *k;      // key (dim,)
  float *v;      // value (dim,)
  float *att;    // buffer for scores/attention values (n_heads, seq_len)
  float *logits; // output logits
  // kv cache
  float *key_cache;   // (layer, seq_len, dim)
  float *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
  Config config;              // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  RunState state;             // buffers for the "wave" of activations in the forward pass
  // some more state needed to properly clean up the memory mapping (sigh)
  int fd;            // file descriptor for memory mapping
  float *data;       // memory mapped data pointer
  ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState *s, Config *p) {
  // we calloc instead of malloc to keep valgrind happy
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  s->x = calloc(p->dim, sizeof(float));
  s->xb = calloc(p->dim, sizeof(float));
  s->xb2 = calloc(p->dim, sizeof(float));
  s->hb = calloc(p->hidden_dim, sizeof(float));
  s->hb2 = calloc(p->hidden_dim, sizeof(float));
  s->q = calloc(p->dim, sizeof(float));
  s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
  s->logits = calloc(p->vocab_size, sizeof(float));
  // ensure all mallocs went fine
  if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
}

void free_run_state(RunState *s) {
  free(s->x);
  free(s->xb);
  free(s->xb2);
  free(s->hb);
  free(s->hb2);
  free(s->q);
  free(s->att);
  free(s->logits);
  free(s->key_cache);
  free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights) {
  int head_size = p->dim / p->n_heads;
  // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
  unsigned long long n_layers = p->n_layers;
  w->token_embedding_table = ptr;
  ptr += p->vocab_size * p->dim;
  w->rms_att_weight = ptr;
  ptr += n_layers * p->dim;
  w->wq = ptr;
  ptr += n_layers * p->dim * (p->n_heads * head_size);
  w->wk = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wv = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wo = ptr;
  ptr += n_layers * (p->n_heads * head_size) * p->dim;
  w->rms_ffn_weight = ptr;
  ptr += n_layers * p->dim;
  w->w1 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->w2 = ptr;
  ptr += n_layers * p->hidden_dim * p->dim;
  w->w3 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->rms_final_weight = ptr;
  ptr += p->dim;
  w->freqs_cos = ptr;
  ptr += p->seq_len * p->n_heads;
  w->freqs_sin = ptr;
  ptr += p->seq_len * p->n_heads;
  w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char *checkpoint, Config *config, TransformerWeights *weights, int *fd, float **data, ssize_t *file_size) {
  FILE *file = fopen(checkpoint, "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  // read in the config header
  if (fread(config, sizeof(Config), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  // negative vocab size is hacky way of signaling unshared weights. bit yikes.
  int shared_weights = config->vocab_size > 0 ? 1 : 0;
  config->vocab_size = abs(config->vocab_size);
  // figure out the file size
#if defined _WIN32
  _fseeki64(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = _ftelli64(file); // get the file size, in bytes
#else
  fseek(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = ftell(file); // get the file size, in bytes
#endif
  fclose(file);
  // memory map the Transformer weights into the data pointer
  *fd = open(checkpoint, O_RDONLY); // open in read only mode
  if (*fd == -1) {
    fprintf(stderr, "open failed!\n");
    exit(EXIT_FAILURE);
  }
  *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) {
    fprintf(stderr, "mmap failed!\n");
    exit(EXIT_FAILURE);
  }
  float *weights_ptr = *data + sizeof(Config) / sizeof(float);
  memory_map_weights(weights, config, weights_ptr, shared_weights);
  fprintf(stderr, "Loaded: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq_len=%d\n",
          config->dim, config->hidden_dim, config->n_layers, config->n_heads,
          config->n_kv_heads, config->vocab_size, config->seq_len);
}

void build_transformer(Transformer *t, char *checkpoint_path) {
  // read in the Config and the Weights from the checkpoint
  read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
  // allocate the RunState buffers
  malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t) {
  // close the memory mapping
  if (t->data != MAP_FAILED) {
    munmap(t->data, t->file_size);
  }
  if (t->fd != -1) {
    close(t->fd);
  }
  // free the RunState buffers
  free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float *restrict o, const float *restrict x, const float *restrict weight, int size) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}

void softmax(float *x, int size) {
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  float inv_sum = 1.0f / sum;
  for (int i = 0; i < size; i++) {
    x[i] *= inv_sum;
  }
}

void matmul(float *xout, float *x, float *w, int n, int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[i * n + j] * x[j];
    }
    xout[i] = val;
  }
}

float *forward(Transformer *transformer, int token, int pos) {

  // a few convenience variables
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim = p->hidden_dim;
  int head_size = dim / p->n_heads;
  const float inv_sqrt_head_size = 1.0f / sqrtf((float)head_size);

  // copy the token embedding into x
  float *content_row = w->token_embedding_table + token * dim;
  memcpy(x, content_row, dim * sizeof(*x));

  // forward all the layers
  for (unsigned long long l = 0; l < p->n_layers; l++) {

    int layer_off = l * dim;
    int layer_kv_off = layer_off * kv_dim;
    int layer_qo_off = layer_off * dim;
    // attention rmsnorm
    rmsnorm(s->xb, x, w->rms_att_weight + layer_off, dim);

    // key and value point to the kv cache
    int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    // qkv matmuls for this position
    float *wq_l = w->wq + layer_qo_off;
    float *wk_l = w->wk + layer_kv_off;
    float *wv_l = w->wv + layer_kv_off;
    float *wo_l = w->wo + layer_qo_off;
    float *w1_l = w->w1 + layer_off * hidden_dim;
    float *w2_l = w->w2 + layer_off * hidden_dim;
    float *w3_l = w->w3 + layer_off * hidden_dim;
    matmul(s->q, s->xb, wq_l, dim, dim);
    matmul(s->k, s->xb, wk_l, dim, kv_dim);
    matmul(s->v, s->xb, wv_l, dim, kv_dim);

    // RoPE: precomputed cosine and sine values for each position/head.
    for (int i = 0; i < p->n_heads; i++) {
      for (int j = 0; j < head_size; j += 2) {
        int dim_idx = j / 2;
        int idx = pos * p->n_heads + dim_idx;
        float fcr = w->freqs_cos[idx];
        float fci = w->freqs_sin[idx];
        float q0 = s->q[i * head_size + j];
        float q1 = s->q[i * head_size + j + 1];
        s->q[i * head_size + j] = q0 * fcr - q1 * fci;
        s->q[i * head_size + j + 1] = q0 * fci + q1 * fcr;
        if (i < p->n_kv_heads) {
          float k0 = s->k[i * head_size + j];
          float k1 = s->k[i * head_size + j + 1];
          s->k[i * head_size + j] = k0 * fcr - k1 * fci;
          s->k[i * head_size + j + 1] = k0 * fci + k1 * fcr;
        }
      }
    }

    // multihead attention. iterate over all heads
    int h;
#pragma omp parallel for private(h)
    for (h = 0; h < p->n_heads; h++) {
      // get the query vector for this head
      float *q = s->q + h * head_size;
      // attention scores for this head
      float *att = s->att + h * p->seq_len;
      int kv_head_off = (h / kv_mul) * head_size;
      // iterate over all timesteps, including the current one
      for (int t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestep
        float *k = s->key_cache + loff + t * kv_dim + kv_head_off;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
          score += q[i] * k[i];
        }
        score *= inv_sqrt_head_size;
        // save the score to the attention buffer
        att[t] = score;
      }

      // softmax the scores to get attention weights, from 0..pos inclusively
      softmax(att, pos + 1);

      // weighted sum of the values, store back into xb
      float *xb = s->xb + h * head_size;
      memset(xb, 0, head_size * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        // get the value vector for this head and at this timestep
        float *v = s->value_cache + loff + t * kv_dim + kv_head_off;
        // get the attention weight for this timestep
        float a = att[t];
        // accumulate the weighted value into xb
        for (int i = 0; i < head_size; i++) {
          xb[i] += a * v[i];
        }
      }
    }

    // final matmul to get the output of the attention
    matmul(s->xb2, s->xb, wo_l, dim, dim);

    // residual connection back into x
    for (int i = 0; i < dim; i++) {
      x[i] += s->xb2[i];
    }

    // ffn rmsnorm
    rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul(s->hb, s->xb, w1_l, dim, hidden_dim);
    matmul(s->hb2, s->xb, w3_l, dim, hidden_dim);

    // SwiGLU non-linearity
    for (int i = 0; i < hidden_dim; i++) {
      float val = s->hb[i];
      // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
      val *= (1.0f / (1.0f + expf(-val)));
      // elementwise multiply with w3(x)
      val *= s->hb2[i];
      s->hb[i] = val;
    }

    // final matmul to get the output of the ffn
    matmul(s->xb, s->hb, w2_l, hidden_dim, dim);

    // residual connection
    for (int i = 0; i < dim; i++) {
      x[i] += s->xb[i];
    }
  }

  // final rmsnorm
  rmsnorm(x, x, w->rms_final_weight, dim);

  // classifier into logits
  matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
  return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
  unsigned char *str;
  int len;
  int id;
} TokenIndex;

typedef struct {
  unsigned char **vocab;
  int *vocab_lens;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  int byte_pieces[256]; // byte -> token id lookup
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
  const TokenIndex *ta = (const TokenIndex *)a;
  const TokenIndex *tb = (const TokenIndex *)b;
  int min_len = ta->len < tb->len ? ta->len : tb->len;
  int cmp = memcmp(ta->str, tb->str, min_len);
  if (cmp != 0) return cmp;
  if (ta->len < tb->len) return -1;
  if (ta->len > tb->len) return 1;
  return 0;
}

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size) {
  // i should have written the vocab_size into the tokenizer file... sigh
  t->vocab_size = vocab_size;
  // malloc space to hold the scores and the strings
  t->vocab = (unsigned char **)malloc(vocab_size * sizeof(unsigned char *));
  t->vocab_lens = (int *)malloc(vocab_size * sizeof(int));
  t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL; // initialized lazily
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i] = -1;
  }
  // read in the file
  FILE *file = fopen(tokenizer_path, "rb");
  if (!file) {
    fprintf(stderr, "couldn't load %s\n", tokenizer_path);
    exit(EXIT_FAILURE);
  }
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
    fprintf(stderr, "failed read\n");
    exit(EXIT_FAILURE);
  }
  int len;
  for (int i = 0; i < vocab_size; i++) {
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    if (fread(&len, sizeof(int), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab_lens[i] = len;
    t->vocab[i] = (unsigned char *)malloc(len + 1);
    if (fread(t->vocab[i], len, 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i][len] = '\0'; // add the string terminating token
  }
  fclose(file);
}

void free_tokenizer(Tokenizer *t) {
  for (int i = 0; i < t->vocab_size; i++) {
    free(t->vocab[i]);
  }
  free(t->vocab);
  free(t->vocab_lens);
  free(t->vocab_scores);
  free(t->sorted_vocab);
}

unsigned char *decode(Tokenizer *t, int prev_token, int token, int *piece_len) {
  (void)prev_token;
  if (token < 0 || token >= t->vocab_size) {
    *piece_len = 0;
    return NULL;
  }
  *piece_len = t->vocab_lens[token];
  return t->vocab[token];
}

void safe_printf(unsigned char *piece, int piece_len) {
  // piece might be a raw byte token, and we only want to print printable chars or whitespace
  // because some of the other bytes can be various control codes, backspace, etc.
  if (piece == NULL || piece_len <= 0) return;
  if (piece_len == 1) {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // bad byte, don't print it
    }
  }
  fwrite(piece, 1, piece_len, stdout);
}

int str_lookup(const unsigned char *str, int str_len, TokenIndex *sorted_vocab, int vocab_size) {
  // efficiently find the perfect match for token bytes in vocab, return its index or -1 if not found
  TokenIndex tok = {.str = (unsigned char *)str, .len = str_len}; // acts as the key to search for
  TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

void init_sorted_vocab(Tokenizer *t) {
  if (t->sorted_vocab != NULL) return;
  t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
  for (int i = 0; i < t->vocab_size; i++) {
    t->sorted_vocab[i].str = t->vocab[i];
    t->sorted_vocab[i].len = t->vocab_lens[i];
    t->sorted_vocab[i].id = i;
  }
  qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  for (int i = 0; i < 256; i++) {
    unsigned char b = (unsigned char)i;
    t->byte_pieces[i] = str_lookup(&b, 1, t->sorted_vocab, t->vocab_size);
  }
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
  // Llama 3.x/3.2 tokenizer is byte-level BPE (tiktoken family).
  // This implementation performs rank-ordered pair merges over raw UTF-8 bytes.
  if (text == NULL) {
    fprintf(stderr, "cannot encode NULL text\n");
    exit(EXIT_FAILURE);
  }
  init_sorted_vocab(t);

  // start at 0 tokens
  *n_tokens = 0;

  // add optional BOS token
  if (bos)
    tokens[(*n_tokens)++] = 128000;

  const unsigned char *piece = (const unsigned char *)text;
  int piece_len = (int)strlen(text);
  if (piece_len > 0) {
    int *ids = malloc(piece_len * sizeof(int));
    int *lens = malloc(piece_len * sizeof(int));
    int *offs = malloc(piece_len * sizeof(int));
    unsigned char *merge_buf = malloc((size_t)piece_len * 2 + 1);
    if (!ids || !lens || !offs || !merge_buf) {
      fprintf(stderr, "malloc failed!\n");
      exit(EXIT_FAILURE);
    }

    int n_parts = 0;
    for (int i = 0; i < piece_len; i++) {
      unsigned char b = piece[i];
      int id = t->byte_pieces[b];
      if (id == -1) {
        id = str_lookup(&b, 1, t->sorted_vocab, t->vocab_size);
      }
      if (id == -1) {
        fprintf(stderr, "tokenizer missing byte token 0x%02X\n", b);
        exit(EXIT_FAILURE);
      }
      ids[n_parts] = id;
      lens[n_parts] = 1;
      offs[n_parts] = i;
      n_parts++;
    }

    while (n_parts > 1) {
      int best_idx = -1;
      int best_id = -1;
      int best_rank = INT_MAX;
      for (int i = 0; i < n_parts - 1; i++) {
        int left_len = lens[i];
        int right_len = lens[i + 1];
        memcpy(merge_buf, piece + offs[i], left_len);
        memcpy(merge_buf + left_len, piece + offs[i + 1], right_len);
        int cand_len = left_len + right_len;
        int id = str_lookup(merge_buf, cand_len, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
          int rank = (int)t->vocab_scores[id];
          if (rank < best_rank) {
            best_rank = rank;
            best_id = id;
            best_idx = i;
          }
        }
      }
      if (best_idx == -1) {
        break;
      }
      ids[best_idx] = best_id;
      lens[best_idx] += lens[best_idx + 1];
      for (int i = best_idx + 1; i < n_parts - 1; i++) {
        ids[i] = ids[i + 1];
        lens[i] = lens[i + 1];
        offs[i] = offs[i + 1];
      }
      n_parts--;
    }

    for (int i = 0; i < n_parts; i++) {
      tokens[(*n_tokens)++] = ids[i];
    }
    free(ids);
    free(lens);
    free(offs);
    free(merge_buf);
  }

  // add optional EOS (=128001) token, if desired
  if (eos)
    tokens[(*n_tokens)++] = 128001;
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
  float prob;
  int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
  int vocab_size;
  ProbIndex *probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
  // return the index that has the highest probability
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

int sample_mult(float *probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b) {
  ProbIndex *a_ = (ProbIndex *)a;
  ProbIndex *b_ = (ProbIndex *)b;
  if (a_->prob > b_->prob)
    return -1;
  if (a_->prob < b_->prob)
    return 1;
  return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  int n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  qsort(probindex, n0, sizeof(ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1; // in case of rounding errors consider all elements
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits) {
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f) {
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocab_size);
  } else {
    // apply the temperature to the logits
    for (int q = 0; q < sampler->vocab_size; q++) {
      logits[q] /= sampler->temperature;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, sampler->vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // simply sample from the predicted probability distribution
      next = sample_mult(logits, sampler->vocab_size, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
    }
  }
  return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
  // return time in milliseconds, for benchmarking the model speed
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
  char *empty_prompt = "";
  if (prompt == NULL) {
    prompt = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  long start = 0;               // used to time our code, only initialized after first iteration
  int next;                     // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence

  while (pos < steps) {

    // forward the transformer to get logits for the next token
    float *logits = forward(transformer, token, pos);

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
    }
    pos++;

    // data-dependent terminating condition: the BOS (=1) token delimits sequences
    if ((next == 128001 || next == 128009) && pos > num_prompt_tokens)
      break;
    // print the token as string, decode it with the Tokenizer object
    int piece_len = 0;
    unsigned char *piece = decode(tokenizer, token, next, &piece_len);
    safe_printf(piece, piece_len); // prints raw token bytes, skipping unsafe single-byte controls
    fflush(stdout);
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) {
      start = time_in_ms();
    }
  }
  printf("\n");

  // report achieved tok/s (pos-1 because the timer starts after first iteration)
  if (pos > 1) {
    long end = time_in_ms();
    fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
  }

  free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
  // read a line from stdin, up to but not including \n
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != NULL) {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
      buffer[len - 1] = '\0'; // strip newline
    }
  }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *cli_user_prompt, char *cli_system_prompt, int steps) {

  // buffers for reading the system prompt and user prompt from stdin
  // you'll notice they are somewhat haphazardly and unsafely set atm
  char *system_prompt = (char *)malloc(32768 * sizeof(char));
  char *user_prompt = (char *)malloc(32768 * sizeof(char));
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc(32768 * sizeof(int));
  int *system_prompt_tokens = (int *)malloc(32768 * sizeof(int));
  int *user_prompt_tokens = (int *)malloc(32768 * sizeof(int));
  int user_idx = 0;

  // start the main loop
  int8_t user_turn = 1; // user starts
  int next;             // will store the next token in the sequence
  int token;            // stores the current token to feed into the transformer

  int pos = 0; // position in the sequence
  while (pos < steps) {

    // when it is the user's turn to contribute tokens to the dialog...
    if (user_turn) {
      // get the (optional) system prompt at position 0
      if (pos == 0) {
        // at position 0, the user can also contribute a system prompt
        prompt_tokens[num_prompt_tokens++] = 128000; // "<|begin_of_text|>"
        prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
        prompt_tokens[num_prompt_tokens++] = 9125;   // "system"
        prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
        prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"
        if (cli_system_prompt == NULL) {
          // system prompt was not passed in, attempt to get it from stdin
          read_stdin("Enter system prompt (optional): ", system_prompt, 32768);
        } else {
          // system prompt was passed in, use it
          strcpy(system_prompt, cli_system_prompt);
        }
        if (system_prompt != NULL) {
          int num_system_prompt_tokens = 0;
          encode(tokenizer, system_prompt, 0, 0, system_prompt_tokens, &num_system_prompt_tokens);
          for (int i = 0; i < num_system_prompt_tokens; i++) {
            prompt_tokens[num_prompt_tokens++] = system_prompt_tokens[i];
          }
        }
        prompt_tokens[num_prompt_tokens++] = 128009; // "<|eot_id|>"
      } else {
        num_prompt_tokens = 0;
      }
      prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
      prompt_tokens[num_prompt_tokens++] = 882;    // "user"
      prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
      prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"
      // get the user prompt
      if (pos == 0 && cli_user_prompt != NULL) {
        // user prompt for position 0 was passed in, use it
        strcpy(user_prompt, cli_user_prompt);
      } else {
        // otherwise get user prompt from stdin
        read_stdin("User (or exit): ", user_prompt, 32768);
        if (strcmp(user_prompt, "exit") == 0)
          break;
      }
      int num_user_prompt_tokens = 0;
      // encode the user prompt into tokens
      encode(tokenizer, user_prompt, 0, 0, user_prompt_tokens, &num_user_prompt_tokens);
      for (int i = 0; i < num_user_prompt_tokens; i++) {
        prompt_tokens[num_prompt_tokens++] = user_prompt_tokens[i];
      }
      prompt_tokens[num_prompt_tokens++] = 128009; // "<|eot_id|>"
      prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
      prompt_tokens[num_prompt_tokens++] = 78191;  // "assistant"
      prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
      prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"

      user_idx = 0; // reset the user index
      user_turn = 0;
      printf("Assistant: ");
    }

    // determine the token to pass into the transformer next
    if (user_idx < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt token
      token = prompt_tokens[user_idx++];
    } else {
      // otherwise use the next token sampled from previous turn
      token = next;
    }
    // EOS (=128009) token ends the Assistant turn
    if (user_idx >= num_prompt_tokens && (token == 128009 || token == 128001)) {
      user_turn = 1;
    }

    // forward the transformer to get logits for the next token
    float *logits = forward(transformer, token, pos);
    next = sample(sampler, logits);
    pos++;

    if (user_idx >= num_prompt_tokens && next != 128009 && next != 128001 && next != 128006) {
      // the Assistant is responding, so print its output
      int piece_len = 0;
      unsigned char *piece = decode(tokenizer, token, next, &piece_len);
      safe_printf(piece, piece_len); // prints raw token bytes, skipping unsafe single-byte controls
      fflush(stdout);
    }
    if (user_idx >= num_prompt_tokens && next == 128009 || next == 128001) {
      printf("\n");
    }
  }
  printf("\n");
  free(prompt_tokens);
  free(system_prompt_tokens);
  free(user_prompt_tokens);
  free(system_prompt);
  free(user_prompt);
}

// ----------------------------------------------------------------------------
// perplexity
// Perplexity score = exp(mean cross-entropy loss)
// ppl_stride == 0: non-overlapping chunks, score second half only.
// ppl_stride > 0: strided overlapping chunks, score last ppl_stride positions per chunk.

void clear_kv_cache(RunState *s, Config *p) {
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  size_t cache_size = (size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float);
  memset(s->key_cache, 0, cache_size);
  memset(s->value_cache, 0, cache_size);
}

float cross_entropy_loss(float *logits, int vocab_size, int target) {
  if (target < 0 || target >= vocab_size)
    return 0.0f;
  float max_val = logits[0];
  for (int i = 1; i < vocab_size; i++) {
    if (logits[i] > max_val)
      max_val = logits[i];
  }
  float sum = 0.0f;
  for (int i = 0; i < vocab_size; i++) {
    sum += expf(logits[i] - max_val);
  }
  float log_sum = max_val + logf(sum);
  return log_sum - logits[target];
}

// Load tokens from file; caller should take care to free
// Can handle either pre-tokenized bin (TOK header) or raw text (C encode; N.B. slow for large files!)
int *load_tokens(const char *data_path, Tokenizer *tokenizer, int *num_tokens_out) {
  FILE *f = fopen(data_path, "rb");
  if (!f) {
    fprintf(stderr, "Could not open %s\n", data_path);
    return NULL;
  }
  char magic[4];
  if (fread(magic, 1, 3, f) != 3) {
    fclose(f);
    fprintf(stderr, "Failed to read file\n");
    return NULL;
  }
  int *tokens = NULL;
  int num_tokens = 0;

  if (strcmp(magic, "TOK") == 0) {
    unsigned int count;
    if (fread(&count, sizeof(count), 1, f) != 1) {
      fclose(f);
      fprintf(stderr, "Failed to read token count\n");
      return NULL;
    }
    tokens = malloc((size_t)count * sizeof(int));
    if (!tokens) {
      fclose(f);
      fprintf(stderr, "malloc failed\n");
      return NULL;
    }
    for (unsigned int i = 0; i < count; i++) {
      unsigned int tok;
      if (fread(&tok, sizeof(tok), 1, f) != 1) {
        free(tokens);
        fclose(f);
        fprintf(stderr, "Failed to read token %u\n", i);
        return NULL;
      }
      tokens[i] = (int)tok;
    }
    fclose(f);
    num_tokens = (int)count;
    fprintf(stderr, "Loaded %d pre-tokenized tokens\n", num_tokens);
  } else {
    fseek(f, 0, SEEK_SET);
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = malloc(fsize + 1);
    if (!text) {
      fclose(f);
      fprintf(stderr, "malloc failed\n");
      return NULL;
    }
    if (fread(text, 1, fsize, f) != (size_t)fsize) {
      free(text);
      fclose(f);
      fprintf(stderr, "Failed to read file\n");
      return NULL;
    }
    text[fsize] = '\0';
    fclose(f);

    int max_tokens = (int)(fsize * 2) + 128;
    tokens = malloc((size_t)max_tokens * sizeof(int));
    if (!tokens) {
      free(text);
      fprintf(stderr, "malloc failed\n");
      return NULL;
    }
    fprintf(stderr, "Tokenising (slow for large files)");
    encode(tokenizer, text, 1, 0, tokens, &num_tokens);
    free(text);
    fprintf(stderr, "Tokenised %d tokens\n", num_tokens);
  }

  *num_tokens_out = num_tokens;
  return tokens;
}

void perplexity(Transformer *transformer, Tokenizer *tokenizer, const char *data_path, int n_ctx, int ppl_stride, int n_chunks) {
  Config *p = &transformer->config;
  RunState *s = &transformer->state;
  int vocab_size = p->vocab_size;

  if (n_ctx > p->seq_len) {
    fprintf(stderr, "n_ctx %d exceeds seq_len %d\n", n_ctx, p->seq_len);
    return;
  }
  if (ppl_stride > 0 && ppl_stride >= n_ctx) {
    fprintf(stderr, "ppl_stride must be < n_ctx when > 0\n");
    return;
  }

  int num_tokens = 0;
  int *tokens = load_tokens(data_path, tokenizer, &num_tokens);
  if (!tokens)
    return;

  if (num_tokens < 2) {
    fprintf(stderr, "Too few tokens in file\n");
    free(tokens);
    return;
  }

  int min_tokens = (ppl_stride == 0) ? (2 * n_ctx) : (n_ctx + 1);
  if (num_tokens < min_tokens) {
    fprintf(stderr, "Need at least %d tokens, got %d. Use longer file or smaller n_ctx.\n",
            min_tokens, num_tokens);
    free(tokens);
    return;
  }

  int stride = (ppl_stride == 0) ? n_ctx : ppl_stride;
  int first_scored = (ppl_stride == 0) ? (n_ctx / 2) : (n_ctx - 1 - ppl_stride);

  int chunk_count = 0;
  double total_loss = 0.0;
  int total_scored = 0;
  #ifdef __FAST_MATH__
    const float inf = 1e10f;  // fake-away infinity check for ffast-math
  #else
    const float inf = INFINITY;
  #endif

  fprintf(stderr, "Calculating perplexity over chunks (n_ctx=%d ppl_stride=%d first_scored=%d)\n",
          n_ctx, ppl_stride, first_scored);

  for (int start = 0; start <= num_tokens - n_ctx; start += stride) {
    if (n_chunks >= 0 && chunk_count >= n_chunks)
      break;

    clear_kv_cache(s, p);

    int *chunk = tokens + start;
    double chunk_loss = 0.0;
    int chunk_scored = 0;

    for (int pos = 0; pos < n_ctx - 1; pos++) {
      float *logits = forward(transformer, chunk[pos], pos);
      int target = chunk[pos + 1];

      if (pos >= first_scored) {
        float loss = cross_entropy_loss(logits, vocab_size, target);
        chunk_loss += (double)loss;
        chunk_scored++;
      }
    }

    if (chunk_scored > 0) {
      total_loss += chunk_loss;
      total_scored += chunk_scored;
      chunk_count++;
      float cum_ppl = (float)exp(total_loss / total_scored);
      fprintf(stderr, "[%d]%.4f ", chunk_count, cum_ppl < 1e2f ? cum_ppl : inf);
    }
  }
  fprintf(stderr, "\n");

  free(tokens);

  if (total_scored == 0) {
    fprintf(stderr, "No chunks scored\n");
    return;
  }

  float mean_loss = (float)(total_loss / total_scored);
  float ppl = (mean_loss < 1e2f) ? (float)exp(mean_loss) : inf;
  fprintf(stdout, "Loss:  %.4f\n", mean_loss);
  fprintf(stdout, "PPL:   %.4f\n", ppl);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
  fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
  fprintf(stderr, "Example: run model.bin -n 4096 -i \"Once upon a time\"\n");
  fprintf(stderr, "         run model.bin -m perplexity -d eval.txt -c 512  // raw text; slow for large files\n");
  fprintf(stderr, "         run model.bin -m perplexity -d eval.tok -c 512  // pre-tokenized bin; faster\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 4096. 0 = max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat|perplexity, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  fprintf(stderr, "  -d <string> data file path (required for perplexity mode)\n");
  fprintf(stderr, "  -c <int>    context size for perplexity, default 512\n");
  fprintf(stderr, "  -S <int>    ppl stride; 0=non-overlapping, >0=strided, default 0\n");
  fprintf(stderr, "  -K <int>    max chunks for perplexity (-1=all), default -1\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

  // default parameters
  char *checkpoint_path = NULL; // e.g. out/model.bin
  char *tokenizer_path = "tokenizer.bin";
  float temperature = 1.0f;        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 1.0f;               // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 4096;                // number of steps to run for
  char *prompt = NULL;             // prompt string
  unsigned long long rng_seed = 0; // seed rng with time by default
  char *mode = "generate";         // generate|chat|perplexity
  char *system_prompt = NULL;      // the (optional) system prompt to use in chat mode
  char *data_path = NULL;          // for perplexity
  int ctx_size = 512;              // for perplexity
  int ppl_stride = 0;              // for perplexity
  int max_chunks = -1;             // for perplexity

  // poor man's C argparse so we can override the defaults above from the command line
  if (argc >= 2) {
    checkpoint_path = argv[1];
  } else {
    error_usage();
  }
  for (int i = 2; i < argc; i += 2) {
    // do some basic validation
    if (i + 1 >= argc) {
      error_usage();
    } // must have arg after flag
    if (argv[i][0] != '-') {
      error_usage();
    } // must start with dash
    if (strlen(argv[i]) != 2) {
      error_usage();
    } // must be -x (one dash, one letter)
    // read in the args
    if (argv[i][1] == 't') {
      temperature = atof(argv[i + 1]);
    } else if (argv[i][1] == 'p') {
      topp = atof(argv[i + 1]);
    } else if (argv[i][1] == 's') {
      rng_seed = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'n') {
      steps = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'i') {
      prompt = argv[i + 1];
    } else if (argv[i][1] == 'z') {
      tokenizer_path = argv[i + 1];
    } else if (argv[i][1] == 'm') {
      mode = argv[i + 1];
    } else if (argv[i][1] == 'y') {
      system_prompt = argv[i + 1];
    } else if (argv[i][1] == 'd') {
      data_path = argv[i + 1];
    } else if (argv[i][1] == 'c') {
      ctx_size = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'S') {
      ppl_stride = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'K') {
      max_chunks = atoi(argv[i + 1]);
    } else {
      error_usage();
    }
  }

  // parameter validation/overrides
  if (rng_seed <= 0)
    rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0)
    temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp)
    topp = 0.9;
  if (steps < 0)
    steps = 0;

  // build the Transformer via the model .bin file
  Transformer transformer;
  build_transformer(&transformer, checkpoint_path);
  if (steps == 0 || steps > transformer.config.seq_len)
    steps = transformer.config.seq_len; // override to ~max length

  // build the Tokenizer via the tokenizer .bin file
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  // run!
  if (strcmp(mode, "perplexity") == 0) {
    if (!data_path) {
      fprintf(stderr, "perplexity mode requires -d <data_file>\n");
      error_usage();
    }
    perplexity(&transformer, &tokenizer, data_path, ctx_size, ppl_stride, max_chunks);
  } else if (strcmp(mode, "generate") == 0) {
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
    free_sampler(&sampler);
  } else if (strcmp(mode, "chat") == 0) {
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);
    chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    free_sampler(&sampler);
  } else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  // memory and file handles cleanup
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);
  return 0;
}
#endif
