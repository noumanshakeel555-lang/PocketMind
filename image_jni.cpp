#include <jni.h>
#include <android/log.h>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <mutex>
#include <thread>

#ifdef POCKETMIND_HAS_SDCPP
#include "stable-diffusion.h"
#endif

#define LOG_TAG "PocketMindImage"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static uint32_t hash_string(const char *s) {
    uint32_t h = 2166136261u;
    if (!s) return h;
    while (*s) {
        h ^= static_cast<uint8_t>(*s++);
        h *= 16777619u;
    }
    return h;
}

static uint8_t clamp_u8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return static_cast<uint8_t>(v);
}

static jintArray make_preview(JNIEnv *env, const char *prompt, const char *model_path, int width, int height, int seed) {
    if (width < 64) width = 64;
    if (height < 64) height = 64;
    if (width > 1024) width = 1024;
    if (height > 1024) height = 1024;

    const uint32_t prompt_hash = hash_string(prompt);
    const uint32_t model_hash = hash_string(model_path);
    const uint32_t mixed = prompt_hash ^ (model_hash << 1) ^ static_cast<uint32_t>(seed * 2654435761u);

    std::vector<jint> pixels(static_cast<size_t>(width) * static_cast<size_t>(height));

    const float cx = width * 0.5f;
    const float cy = height * 0.5f;
    const float max_dim = static_cast<float>(width > height ? width : height);

    const int base_r = 70 + (mixed & 0x7F);
    const int base_g = 50 + ((mixed >> 8) & 0x7F);
    const int base_b = 90 + ((mixed >> 16) & 0x7F);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float dx = (x - cx) / max_dim;
            const float dy = (y - cy) / max_dim;
            const float dist = std::sqrt(dx * dx + dy * dy);
            const float wave = std::sin((x * 0.035f) + (y * 0.021f) + (mixed % 1000) * 0.01f);
            const float ring = std::sin((dist * 42.0f) - (mixed % 360) * 0.017f);

            int r = base_r + static_cast<int>(70.0f * wave) + static_cast<int>(45.0f * (1.0f - dist));
            int g = base_g + static_cast<int>(55.0f * ring) + static_cast<int>(35.0f * (static_cast<float>(x) / width));
            int b = base_b + static_cast<int>(80.0f * std::sin((x + y) * 0.018f)) + static_cast<int>(45.0f * (static_cast<float>(y) / height));

            if (((x / 32 + y / 32 + (mixed & 7)) % 9) == 0) {
                r += 20; g += 20; b += 25;
            }

            const uint8_t rr = clamp_u8(r);
            const uint8_t gg = clamp_u8(g);
            const uint8_t bb = clamp_u8(b);
            pixels[static_cast<size_t>(y) * width + x] = static_cast<jint>(0xFF000000u | (rr << 16) | (gg << 8) | bb);
        }
    }

    jintArray arr = env->NewIntArray(width * height);
    if (!arr) return nullptr;
    env->SetIntArrayRegion(arr, 0, width * height, pixels.data());
    return arr;
}

static std::string g_last_error;

#ifdef POCKETMIND_HAS_SDCPP
static std::mutex g_sd_mutex;
static sd_ctx_t *g_sd_ctx = nullptr;
static std::string g_sd_model_path;

static void sd_log_bridge(enum sd_log_level_t level, const char *text, void *) {
    if (!text) return;
    if (level == SD_LOG_ERROR) {
        g_last_error = text;
        LOGE("%s", text);
    } else {
        LOGI("%s", text);
    }
}

static bool ensure_sd_context(const char *model_path) {
    if (!model_path || model_path[0] == '\0') {
        g_last_error = "No Stable Diffusion model path provided";
        return false;
    }

    if (g_sd_ctx && g_sd_model_path == model_path) {
        return true;
    }

    if (g_sd_ctx) {
        free_sd_ctx(g_sd_ctx);
        g_sd_ctx = nullptr;
        g_sd_model_path.clear();
    }

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path = model_path;
    ctx_params.n_threads = std::max(2u, std::thread::hardware_concurrency());
    ctx_params.enable_mmap = true;
    ctx_params.flash_attn = false;

    // CPU first for stability. Once this works, we can enable backend="vulkan" in a separate phase.
    ctx_params.backend = "cpu";
    ctx_params.params_backend = "cpu";

    sd_set_log_callback(sd_log_bridge, nullptr);

    LOGI("Loading stable-diffusion.cpp model: %s", model_path);
    g_sd_ctx = new_sd_ctx(&ctx_params);
    if (!g_sd_ctx) {
        if (g_last_error.empty()) g_last_error = "Failed to load Stable Diffusion model";
        LOGE("new_sd_ctx failed");
        return false;
    }

    if (!sd_ctx_supports_image_generation(g_sd_ctx)) {
        free_sd_ctx(g_sd_ctx);
        g_sd_ctx = nullptr;
        g_last_error = "This model does not support image generation";
        return false;
    }

    g_sd_model_path = model_path;
    g_last_error.clear();
    return true;
}
#endif

extern "C" JNIEXPORT jintArray JNICALL
Java_com_arslan_llamachat_image_ImageEngine_generatePreviewNative(
        JNIEnv *env,
        jobject,
        jstring prompt_j,
        jstring model_path_j,
        jint width,
        jint height,
        jint seed) {

    const char *prompt = prompt_j ? env->GetStringUTFChars(prompt_j, nullptr) : "";
    const char *model_path = model_path_j ? env->GetStringUTFChars(model_path_j, nullptr) : "";

    jintArray out = make_preview(env, prompt, model_path, width, height, seed);

    if (prompt_j) env->ReleaseStringUTFChars(prompt_j, prompt);
    if (model_path_j) env->ReleaseStringUTFChars(model_path_j, model_path);

    LOGI("Generated safe native preview image %dx%d", width, height);
    return out;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_arslan_llamachat_image_ImageEngine_generateStableDiffusionNative(
        JNIEnv *env,
        jobject,
        jstring prompt_j,
        jstring negative_prompt_j,
        jstring model_path_j,
        jint width,
        jint height,
        jint steps,
        jfloat cfg_scale,
        jint seed) {

    const char *prompt = prompt_j ? env->GetStringUTFChars(prompt_j, nullptr) : "";
    const char *negative_prompt = negative_prompt_j ? env->GetStringUTFChars(negative_prompt_j, nullptr) : "";
    const char *model_path = model_path_j ? env->GetStringUTFChars(model_path_j, nullptr) : "";

    jintArray result = nullptr;
    g_last_error.clear();

#ifdef POCKETMIND_HAS_SDCPP
    {
        std::lock_guard<std::mutex> lock(g_sd_mutex);

        width = std::max(128, std::min(768, static_cast<int>(width)));
        height = std::max(128, std::min(768, static_cast<int>(height)));
        steps = std::max(1, std::min(50, static_cast<int>(steps)));
        cfg_scale = std::max(1.0f, std::min(20.0f, static_cast<float>(cfg_scale)));

        if (!ensure_sd_context(model_path)) {
            LOGE("SD context unavailable: %s", g_last_error.c_str());
            result = nullptr;
        } else {
            sd_img_gen_params_t gen_params;
            sd_img_gen_params_init(&gen_params);
            gen_params.prompt = prompt;
            gen_params.negative_prompt = negative_prompt;
            gen_params.width = width;
            gen_params.height = height;
            gen_params.seed = seed <= 0 ? 42 : seed;
            gen_params.batch_count = 1;
            gen_params.clip_skip = -1;
            // Keep generation parameters compatible across stable-diffusion.cpp revisions.
            // Some revisions do not expose guidance.txt_cfg or vae_tiling_params in this struct.
            gen_params.sample_params.sample_steps = steps;
            gen_params.sample_params.sample_method = sd_get_default_sample_method(g_sd_ctx);
            gen_params.sample_params.scheduler = sd_get_default_scheduler(g_sd_ctx, gen_params.sample_params.sample_method);

            LOGI("Running real Stable Diffusion: %dx%d steps=%d cfg=%.2f", width, height, steps, cfg_scale);
            sd_image_t *images = generate_image(g_sd_ctx, &gen_params);
            if (!images || !images[0].data) {
                if (g_last_error.empty()) g_last_error = "Stable Diffusion generation returned no image";
                LOGE("generate_image failed");
                if (images) std::free(images);
                result = nullptr;
            } else {
                const int out_w = static_cast<int>(images[0].width);
                const int out_h = static_cast<int>(images[0].height);
                const int channels = static_cast<int>(images[0].channel);
                std::vector<jint> pixels(static_cast<size_t>(out_w) * static_cast<size_t>(out_h));

                for (int y = 0; y < out_h; ++y) {
                    for (int x = 0; x < out_w; ++x) {
                        const size_t idx = static_cast<size_t>(y) * out_w + x;
                        const size_t src = idx * channels;
                        const uint8_t r = images[0].data[src + 0];
                        const uint8_t g = channels > 1 ? images[0].data[src + 1] : r;
                        const uint8_t b = channels > 2 ? images[0].data[src + 2] : r;
                        const uint8_t a = channels > 3 ? images[0].data[src + 3] : 255;
                        pixels[idx] = static_cast<jint>((a << 24) | (r << 16) | (g << 8) | b);
                    }
                }

                result = env->NewIntArray(out_w * out_h);
                if (result) env->SetIntArrayRegion(result, 0, out_w * out_h, pixels.data());

                std::free(images[0].data);
                std::free(images);
                LOGI("Real Stable Diffusion image generated successfully");
            }
        }
    }
#else
    // Safe fallback if stable-diffusion.cpp source has not been added yet.
    g_last_error = "Stable Diffusion source is not enabled in this build; generated a safe native preview instead.";
    result = make_preview(env, prompt, model_path, width, height, seed);
#endif

    if (prompt_j) env->ReleaseStringUTFChars(prompt_j, prompt);
    if (negative_prompt_j) env->ReleaseStringUTFChars(negative_prompt_j, negative_prompt);
    if (model_path_j) env->ReleaseStringUTFChars(model_path_j, model_path);

    return result;
}


extern "C" JNIEXPORT jstring JNICALL
Java_com_arslan_llamachat_image_ImageEngine_lastImageErrorNative(JNIEnv *env, jobject) {
    return env->NewStringUTF(g_last_error.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arslan_llamachat_image_ImageEngine_engineInfoNative(JNIEnv *env, jobject) {
#ifdef POCKETMIND_HAS_SDCPP
    std::string info = "Real Stable Diffusion engine ready";
    const char *version = sd_version();
    if (version && version[0] != '\0') info += std::string(" · ") + version;
    info += " · CPU safe mode";
    return env->NewStringUTF(info.c_str());
#else
    return env->NewStringUTF("Native image bridge ready. Add stable-diffusion.cpp source folder to enable real offline generation.");
#endif
}

extern "C" JNIEXPORT void JNICALL
Java_com_arslan_llamachat_image_ImageEngine_freeImageEngineNative(JNIEnv *, jobject) {
#ifdef POCKETMIND_HAS_SDCPP
    std::lock_guard<std::mutex> lock(g_sd_mutex);
    if (g_sd_ctx) {
        free_sd_ctx(g_sd_ctx);
        g_sd_ctx = nullptr;
        g_sd_model_path.clear();
    }
#endif
}
