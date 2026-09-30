#include <jni.h>
#include <android/log.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <random>
std::random_device rd;

#include "llama.h"

#define LOG_TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

std::once_flag g_backend_once;
std::atomic<bool> g_stop{false};

struct State {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    llama_sampler * smpl  = nullptr;
    int32_t n_ctx = 2048;
};

State g_state;

std::string jstr(JNIEnv *env, jstring js) {
    if (!js) return {};

    const char *c = env->GetStringUTFChars(js, nullptr);
    std::string s(c ? c : "");

    if (c) {
        env->ReleaseStringUTFChars(js, c);
    }

    return s;
}

struct SamplerParams {
    float temperature = 0.7f;
    float top_p = 0.9f;
    int32_t top_k = 40;
    float repeat_penalty = 1.1f;
    int32_t n_prev = 64;
};

SamplerParams readSampler(JNIEnv *env, jobject jparams) {

    SamplerParams out;

    if (!env || !jparams)
        return out;

    jclass cls = env->GetObjectClass(jparams);

    if (!cls)
        return out;

    auto readFloat =
        [&](const char *name, float &target) {

            jfieldID fid =
                env->GetFieldID(cls, name, "F");

            if (fid)
                target = env->GetFloatField(jparams, fid);
        };

    auto readInt =
        [&](const char *name, int32_t &target) {

            jfieldID fid =
                env->GetFieldID(cls, name, "I");

            if (fid)
                target = env->GetIntField(jparams, fid);
        };

    readFloat("temperature", out.temperature);
    readFloat("topP", out.top_p);
    readFloat("repeatPenalty", out.repeat_penalty);

    readInt("topK", out.top_k);
    readInt("nPrev", out.n_prev);

    env->DeleteLocalRef(cls);

    return out;
}


/*
 * Add one token to llama_batch.
 *
 * This replaces the old common_batch_add().
 */
void batch_add(
        llama_batch &batch,
        llama_token token,
        llama_pos pos,
        llama_seq_id seq_id,
        bool need_logits) {

    const int i = batch.n_tokens;

    batch.token[i] = token;

    batch.pos[i] = pos;

    batch.n_seq_id[i] = 1;

    batch.seq_id[i][0] = seq_id;

    batch.logits[i] = need_logits ? 1 : 0;

    batch.n_tokens++;
}


/*
 * Clear a batch without llama_batch_clear(),
 * because that function does not exist in your version.
 */
void batch_clear(llama_batch &batch) {
    batch.n_tokens = 0;
}

} // namespace


extern "C" {


JNIEXPORT void JNICALL
Java_com_example_llamachat_inference_LlamaEngine_nativeInit(
        JNIEnv *,
        jclass) {

    std::call_once(g_backend_once, []() {

        llama_backend_init();

        LOGI("llama backend initialized");
    });
}


JNIEXPORT jboolean JNICALL
Java_com_example_llamachat_inference_LlamaEngine_nativeLoad(
        JNIEnv *env,
        jclass,
        jstring jpath,
        jint jn_ctx,
        jint jn_threads,
        jint jn_gpu_layers) {

    /*
     * Free previous model.
     */

    if (g_state.smpl) {

        llama_sampler_free(g_state.smpl);
        g_state.smpl = nullptr;
    }

    if (g_state.ctx) {

        llama_free(g_state.ctx);
        g_state.ctx = nullptr;
    }

    if (g_state.model) {

        llama_model_free(g_state.model);
        g_state.model = nullptr;
    }


    std::string path = jstr(env, jpath);

    if (path.empty()) {

        LOGE("nativeLoad: empty model path");

        return JNI_FALSE;
    }


    /*
     * Model parameters.
     */

    llama_model_params mp =
        llama_model_default_params();

    mp.n_gpu_layers =
        static_cast<int32_t>(jn_gpu_layers);


    /*
     * Load GGUF model.
     */

    g_state.model =
        llama_model_load_from_file(
            path.c_str(),
            mp);

    if (!g_state.model) {

        LOGE(
            "llama_model_load_from_file failed: %s",
            path.c_str());

        return JNI_FALSE;
    }


    /*
     * Context parameters.
     */

    g_state.n_ctx =
        static_cast<int32_t>(jn_ctx);

    llama_context_params cp =
        llama_context_default_params();

    cp.n_ctx =
        g_state.n_ctx;

    cp.n_threads =
        static_cast<int32_t>(jn_threads);

    cp.n_threads_batch =
        static_cast<int32_t>(jn_threads);

    cp.n_batch = 512;

    cp.n_ubatch = 512;

    cp.n_seq_max = 1;

    cp.type_k = GGML_TYPE_F16;

    cp.type_v = GGML_TYPE_F16;


    /*
     * Newer llama.cpp still supports this function,
     * although it is deprecated.
     */
    g_state.ctx =
        llama_new_context_with_model(
            g_state.model,
            cp);

    if (!g_state.ctx) {

        LOGE(
            "llama_new_context_with_model failed");

        llama_model_free(g_state.model);

        g_state.model = nullptr;

        return JNI_FALSE;
    }


    LOGI(
        "Model loaded: %s | ctx=%d | threads=%d | gpu_layers=%d",
        path.c_str(),
        g_state.n_ctx,
        static_cast<int>(jn_threads),
        static_cast<int>(jn_gpu_layers));

    return JNI_TRUE;
}


JNIEXPORT void JNICALL
Java_com_example_llamachat_inference_LlamaEngine_nativeUnload(
        JNIEnv *,
        jclass) {

    if (g_state.smpl) {

        llama_sampler_free(g_state.smpl);

        g_state.smpl = nullptr;
    }

    if (g_state.ctx) {

        llama_free(g_state.ctx);

        g_state.ctx = nullptr;
    }

    if (g_state.model) {

        llama_model_free(g_state.model);

        g_state.model = nullptr;
    }

    LOGI("Model unloaded");
}


JNIEXPORT void JNICALL
Java_com_example_llamachat_inference_LlamaEngine_nativeStop(
        JNIEnv *,
        jclass) {

    g_stop.store(true);
}


/*
 * Chat template.
 *
 * IMPORTANT:
 *
 * llama_chat_apply_template() expects:
 *
 *   tmpl
 *   llama_chat_message[]
 *   number of messages
 *   add_assistant
 *   buffer
 *   buffer_size
 *
 * It does NOT accept JSON.
 *
 * Therefore this function currently treats the incoming
 * string as an already formatted prompt.
 *
 * This is safer until Kotlin passes structured messages.
 */
JNIEXPORT jstring JNICALL
Java_com_example_llamachat_inference_LlamaEngine_nativeApplyChatTemplate(
        JNIEnv *env,
        jclass,
        jstring jmessages_json,
        jboolean jadd_assistant) {

    std::string body =
        jstr(env, jmessages_json);

    if (!g_state.model) {

        return env->NewStringUTF("");
    }


    /*
     * For now return the already formatted prompt.
     *
     * Llama-3.2 uses:
     *
     * <|begin_of_text|>
     * <|start_header_id|>user<|end_header_id|>
     * ...
     *
     * If Kotlin already generates this, don't apply
     * the template again.
     */

    std::string out = body;

    if (static_cast<bool>(jadd_assistant)) {

        out +=
            "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    return env->NewStringUTF(out.c_str());
}


/*
 * Generate.
 */
JNIEXPORT jint JNICALL
Java_com_example_llamachat_inference_LlamaEngine_nativeGenerate(
        JNIEnv *env,
        jclass,
        jstring jprompt,
        jobject jsampler,
        jint jmax_tokens,
        jobject jcallback) {

    if (!g_state.ctx ||
        !g_state.model) {

        LOGE(
            "nativeGenerate: model not loaded");

        return -1;
    }

    if (!jcallback) {

        LOGE(
            "nativeGenerate: null callback");

        return -1;
    }

    g_stop.store(false);


    /*
     * Clear the KV cache for this sequence.
     *
     * Kotlin's buildPrompt() replays the full conversation history
     * every turn, so there is no reason to keep KV cells from a
     * previous turn alive. Without this clear, the second turn's
     * prompt tokens land at positions 0..N-1 — which the previous
     * turn already wrote to — and llama_decode() rejects the batch
     * with "llama_decode(prompt) failed".
     *
     * Replaying from a clean cache matches the official llama.cpp
     * chat example for single-sequence chat loops.
     */
    llama_memory_clear(llama_get_memory(g_state.ctx), true);


    /*
     * Sampler parameters.
     */

    SamplerParams sp =
        readSampler(env, jsampler);


    /*
     * Create sampler.
     */

    if (g_state.smpl) {

        llama_sampler_free(g_state.smpl);

        g_state.smpl = nullptr;
    }

    g_state.smpl =
        llama_sampler_chain_init(
            llama_sampler_chain_default_params());


    llama_sampler_chain_add(
        g_state.smpl,
        llama_sampler_init_top_k(sp.top_k));


    llama_sampler_chain_add(
        g_state.smpl,
        llama_sampler_init_top_p(
            sp.top_p,
            1));


    llama_sampler_chain_add(
        g_state.smpl,
        llama_sampler_init_temp(
            sp.temperature));


    llama_sampler_chain_add(
        g_state.smpl,
        llama_sampler_init_dist(
            rd()));


    /*
     * Get vocabulary.
     *
     * NEW API:
     *
     * llama_tokenize() requires llama_vocab,
     * not llama_model.
     */

    const llama_vocab *vocab =
        llama_model_get_vocab(
            g_state.model);

    if (!vocab) {

        LOGE("Could not get vocabulary");

        return -1;
    }


    /*
     * Tokenize.
     */

    std::string prompt =
        jstr(env, jprompt);

    int32_t prompt_len =
        static_cast<int32_t>(prompt.size());


    /*
     * First ask tokenizer how many tokens
     * are required.
     *
     * llama_tokenize() uses a sentinel convention for the "count"
     * query: when n_tokens_max is too small to hold the result, the
     * call returns -res.size() so callers can allocate the right
     * buffer. Passing n_tokens_max=0 always triggers this branch,
     * even on a fully valid prompt.
     *
     * Therefore a negative return here is NOT a tokenizer error
     * when we're counting — it is the token count. The official
     * llama.cpp example handles it the same way:
     *
     *   const int n = -llama_tokenize(vocab, ..., NULL, 0, true, true);
     *
     * Genuine failure is signalled only by INT32_MIN.
     */

    int n_prompt =
        llama_tokenize(
            vocab,
            prompt.c_str(),
            prompt_len,
            nullptr,
            0,
            false,
            true);


    if (n_prompt == INT32_MIN) {

        LOGE(
            "Failed to determine token count: "
            "prompt overflow (INT32_MIN); prompt_len=%d",
            prompt_len);

        return -1;
    }

    if (n_prompt < 0) {

        /*
         * Counting call: -rc is the count of tokens the real
         * call will produce.
         */
        n_prompt = -n_prompt;
    }


    if (n_prompt == 0) {

        LOGE(
            "Failed to determine token count: "
            "tokenizer produced 0 tokens; prompt_len=%d, "
            "head=\"%.64s%s\"",
            prompt_len,
            prompt.c_str(),
            prompt_len > 64 ? "..." : "");

        return -1;
    }


    std::vector<llama_token> prompt_tokens(
        n_prompt);


    /*
     * Actual tokenization.
     */

    n_prompt =
        llama_tokenize(
            vocab,
            prompt.c_str(),
            prompt_len,
            prompt_tokens.data(),
            n_prompt,
            false,
            true);


    if (n_prompt <= 0) {

        LOGE("Tokenization failed");

        return -1;
    }


    prompt_tokens.resize(n_prompt);


    /*
     * Context handling.
     */

    int32_t ctx_size =
        llama_n_ctx(g_state.ctx);

    int32_t max_new =
        static_cast<int32_t>(jmax_tokens);

    if (max_new <= 0)
        max_new = 256;


    if (max_new > ctx_size - 64)
        max_new = ctx_size - 64;


    /*
     * If prompt is too long,
     * remove tokens from the left.
     */

    if (n_prompt + max_new > ctx_size) {

        int32_t drop =
            n_prompt + max_new - ctx_size;

        if (drop >= n_prompt) {

            LOGE(
                "Prompt is too large for context");

            return -1;
        }

        prompt_tokens.erase(
            prompt_tokens.begin(),
            prompt_tokens.begin() + drop);

        n_prompt -= drop;
    }


    /*
     * Allocate batch.
     *
     * We need capacity for prompt + generated tokens.
     */

    llama_batch batch =
        llama_batch_init(
            ctx_size,
            0,
            1);


    /*
     * Add prompt tokens.
     */

    for (int i = 0; i < n_prompt; ++i) {

        batch_add(
            batch,
            prompt_tokens[i],
            i,
            0,
            i == n_prompt - 1);
    }


    /*
     * Decode prompt.
     */

    if (llama_decode(
            g_state.ctx,
            batch) != 0) {

        LOGE(
            "llama_decode(prompt) failed");

        llama_batch_free(batch);

        return -1;
    }


    int32_t n_decoded =
        n_prompt;


    /*
     * Find Kotlin callback.
     *
     * Kotlin:
     *
     * fun onToken(
     *     token: String,
     *     cumulative: Int
     * )
     */

    jclass cb_cls =
        env->GetObjectClass(jcallback);

    jmethodID on_token =
        env->GetMethodID(
            cb_cls,
            "onToken",
            "(Ljava/lang/String;I)V");


    if (!on_token) {

        LOGE(
            "Could not find callback.onToken");

        env->DeleteLocalRef(cb_cls);

        llama_batch_free(batch);

        return -1;
    }

    env->DeleteLocalRef(cb_cls);


    /*
     * Generation loop.
     */

    int32_t generated = 0;

    std::string piece;

    piece.reserve(64);


    while (generated < max_new) {

        if (g_stop.load())
            break;


        /*
         * Sample next token.
         */

        llama_token id =
            llama_sampler_sample(
                g_state.smpl,
                g_state.ctx,
                -1);


        llama_sampler_accept(
            g_state.smpl,
            id);


        /*
         * Stop on EOS/EOT.
         */

        if (llama_vocab_is_eog(
                vocab,
                id)) {

            break;
        }


        /*
         * Convert token -> text.
         */

        char token_buf[256];

        int n =
            llama_token_to_piece(
                vocab,
                id,
                token_buf,
                sizeof(token_buf),
                0,
                false);


        if (n > 0) {

            piece.assign(
                token_buf,
                token_buf + n);


            jstring jpiece =
                env->NewStringUTF(
                    piece.c_str());


            env->CallVoidMethod(
                jcallback,
                on_token,
                jpiece,
                generated + 1);


            env->DeleteLocalRef(
                jpiece);


            if (env->ExceptionCheck()) {

                LOGE(
                    "Java callback threw");

                env->ExceptionClear();

                break;
            }
        }


        /*
         * Clear batch.
         *
         * New API has no llama_batch_clear().
         */

        batch_clear(batch);


        /*
         * Add generated token.
         */

        batch_add(
            batch,
            id,
            n_decoded,
            0,
            true);


        n_decoded++;


        /*
         * Decode generated token.
         */

        if (llama_decode(
                g_state.ctx,
                batch) != 0) {

            LOGE(
                "llama_decode(step %d) failed",
                generated);

            break;
        }


        generated++;
    }


    llama_batch_free(batch);


    LOGI(
        "Generated %d tokens | stopped=%s",
        generated,
        g_stop.load() ? "yes" : "no");


    return static_cast<jint>(
        generated);
}


} // extern "C"