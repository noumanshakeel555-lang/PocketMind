#include <cstdio>
#include <algorithm>
#include <jni.h>
#include <string>
#include <vector>
#include <atomic>
#include <cstdint>
#include <thread>
#include <chrono>
#include <android/log.h>
#include "llama.h"
#include "ggml.h"

#define TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static llama_model*   g_model     = nullptr;
static llama_context* g_ctx       = nullptr;
static int            g_n_ctx     = 2048;
static int            g_n_threads = 4;
static int            g_n_batch   = 512;
static int            g_n_gpu     = 0;
static std::atomic<bool> g_stop(false);
static std::atomic<bool> g_busy(false);

static const uint32_t MY_GGUF_MAGIC = 0x46554747u;

static bool validate_gguf(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    uint32_t magic = 0;
    bool ok = fread(&magic, 4, 1, f) == 1 && magic == MY_GGUF_MAGIC;
    fclose(f);
    return ok;
}

static void free_ctx() {
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
}

static void free_all() {
    free_ctx();
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
}

static bool make_ctx() {
    free_ctx();
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t)g_n_ctx;
    cp.n_batch         = (uint32_t)g_n_batch;
    cp.n_ubatch        = (uint32_t)g_n_batch;
    cp.n_threads       = (uint32_t)g_n_threads;
    cp.n_threads_batch = (uint32_t)g_n_threads;
    cp.type_k          = GGML_TYPE_F16;
    cp.type_v          = GGML_TYPE_F16;
    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx && g_n_ctx > 512) {
        LOGI("Retrying at 512 ctx");
        g_n_ctx = 512;
        cp.n_ctx = 512;
        g_ctx = llama_init_from_model(g_model, cp);
    }
    return g_ctx != nullptr;
}

static int auto_threads() {
    int c = (int)std::thread::hardware_concurrency();
    return c > 0 ? c : 4;
}

static bool has_vulkan() {
#ifdef GGML_USE_VULKAN
    return true;
#else
    return false;
#endif
}

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_arslan_llamachat_llm_LlamaEngine_loadModel(
        JNIEnv* env, jobject,
        jstring jpath, jint n_ctx, jint n_threads, jint n_batch, jint n_gpu_layers) {

    if (g_busy.load()) return env->NewStringUTF("BUSY: generation in progress");

    const char* path = env->GetStringUTFChars(jpath, nullptr);

    if (!validate_gguf(path)) {
        env->ReleaseStringUTFChars(jpath, path);
        return env->NewStringUTF("CORRUPT: File is not valid GGUF. Delete and re-download.");
    }

    free_all();
    g_n_ctx     = (int)n_ctx;
    g_n_threads = (n_threads <= 0) ? auto_threads() : (int)n_threads;
    g_n_batch   = (n_batch   <= 0) ? 512            : (int)n_batch;
    g_n_gpu     = (int)n_gpu_layers;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = g_n_gpu;
    mp.use_mmap     = true;
    mp.use_mlock    = false;

    g_model = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);

    if (!g_model) return env->NewStringUTF("LOAD_FAIL: Could not read model weights.");
    if (!make_ctx()) {
        llama_model_free(g_model); g_model = nullptr;
        return env->NewStringUTF("OOM: Not enough RAM to create context.");
    }

    bool gpu_on = g_n_gpu != 0 && has_vulkan();
    LOGI("Model ready. ctx=%d threads=%d batch=%d gpu_layers=%d vulkan=%d",
         g_n_ctx, g_n_threads, g_n_batch, g_n_gpu, (int)has_vulkan());

    std::string info = std::string("ok|threads=") + std::to_string(g_n_threads)
        + "|gpu=" + (gpu_on ? "vulkan" : "cpu")
        + "|ctx=" + std::to_string(g_n_ctx);
    return env->NewStringUTF(info.c_str());
}

JNIEXPORT jstring JNICALL
Java_com_arslan_llamachat_llm_LlamaEngine_setContextSize(
        JNIEnv* env, jobject, jint n_ctx) {
    if (!g_model) return env->NewStringUTF("No model loaded");
    if (g_busy.load()) return env->NewStringUTF("BUSY");
    g_n_ctx = (int)n_ctx;
    return make_ctx() ? env->NewStringUTF("ok") : env->NewStringUTF("Failed");
}

JNIEXPORT jboolean JNICALL
Java_com_arslan_llamachat_llm_LlamaEngine_isVulkanAvailable(JNIEnv*, jobject) {
    return (jboolean)has_vulkan();
}

JNIEXPORT jint JNICALL
Java_com_arslan_llamachat_llm_LlamaEngine_getOptimalThreads(JNIEnv*, jobject) {
    return (jint)auto_threads();
}

JNIEXPORT jint JNICALL
Java_com_arslan_llamachat_llm_LlamaEngine_getContextSize(JNIEnv*, jobject) {
    return (jint)g_n_ctx;
}

JNIEXPORT void JNICALL
Java_com_arslan_llamachat_llm_LlamaEngine_generate(
        JNIEnv* env, jobject,
        jstring jprompt, jint max_tokens,
        jfloat temperature, jint top_k, jfloat top_p,
        jobject callback) {

    if (!g_model || !g_ctx) { LOGE("Not initialised"); return; }
    if (g_busy.exchange(true)) { LOGE("Already generating"); return; }
    g_stop = false;

    const char* ps = env->GetStringUTFChars(jprompt, nullptr);
    std::string prompt(ps);
    env->ReleaseStringUTFChars(jprompt, ps);

    const llama_vocab* vocab = llama_model_get_vocab(g_model);

    std::vector<llama_token> tokens(prompt.size() + 32);
    int n_tok = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                               tokens.data(), (int32_t)tokens.size(), true, true);
    if (n_tok < 0) {
        tokens.resize(-n_tok);
        n_tok = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                               tokens.data(), (int32_t)tokens.size(), true, true);
    }
    tokens.resize(n_tok);

    if (n_tok >= g_n_ctx) {
        int keep = g_n_ctx - 64;
        tokens.erase(tokens.begin(), tokens.begin() + (n_tok - keep));
        n_tok = (int)tokens.size();
        LOGI("Prompt truncated to %d tokens", n_tok);
    }

    if (!make_ctx()) {
        LOGE("Context reset failed");
        g_busy = false;
        return;
    }

    auto t0 = std::chrono::steady_clock::now();
    int n_processed = 0;
    while (n_processed < n_tok && !g_stop) {
        int end = std::min(n_processed + g_n_batch, n_tok);
        std::vector<llama_token> chunk(tokens.begin() + n_processed, tokens.begin() + end);
        llama_batch batch = llama_batch_get_one(chunk.data(), (int32_t)chunk.size());
        if (llama_decode(g_ctx, batch) != 0) {
            LOGE("Decode failed at %d", n_processed);
            g_busy = false;
            return;
        }
        n_processed = end;
    }
    auto t1 = std::chrono::steady_clock::now();
    float prompt_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    LOGI("Prompt: %d tok in %.0fms = %.1f tok/s", n_tok, prompt_ms, n_tok * 1000.0f / prompt_ms);

    jclass    cls    = env->GetObjectClass(callback);
    jmethodID onTok  = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");
    jmethodID onDone = env->GetMethodID(cls, "onDone",  "()V");

    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k((int32_t)top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p((float)top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp((float)temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    char piece[256];
    int  n_gen = 0;
    auto t2 = std::chrono::steady_clock::now();

    for (int i = 0; i < max_tokens && !g_stop; i++) {
        llama_token tok = llama_sampler_sample(sampler, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        int n = llama_token_to_piece(vocab, tok, piece, (int)sizeof(piece) - 1, 0, true);
        if (n > 0) {
            piece[n] = '\0';
            jstring js = env->NewStringUTF(piece);
            env->CallVoidMethod(callback, onTok, js);
            env->DeleteLocalRef(js);
        }
        llama_batch next = llama_batch_get_one(&tok, 1);
        if (llama_decode(g_ctx, next) != 0) break;
        n_gen++;
    }

    auto t3 = std::chrono::steady_clock::now();
    float gen_ms = std::chrono::duration<float, std::milli>(t3 - t2).count();
    if (n_gen > 0) LOGI("Gen: %d tok in %.0fms = %.1f tok/s", n_gen, gen_ms, n_gen * 1000.0f / gen_ms);

    llama_sampler_free(sampler);
    g_busy = false;
    env->CallVoidMethod(callback, onDone);
}

JNIEXPORT void JNICALL
Java_com_arslan_llamachat_llm_LlamaEngine_stop(JNIEnv*, jobject) { g_stop = true; }

JNIEXPORT void JNICALL
Java_com_arslan_llamachat_llm_LlamaEngine_freeModel(JNIEnv*, jobject) {
    g_stop = true;
    int w = 0;
    while (g_busy.load() && w++ < 100)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    free_all();
}

}
