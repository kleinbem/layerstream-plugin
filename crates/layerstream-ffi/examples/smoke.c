#include "layerstream.h"
#include <stdio.h>

static bool print_token(const char *chunk, size_t len, void *user_data) {
    (void)user_data;
    fwrite(chunk, 1, len, stdout);
    fflush(stdout);
    return true;
}

int main(int argc, char **argv) {
    printf("layerstream %s\n", layerstream_version());
    if (argc < 2) {
        return 0;
    }

    LayerStreamEngine *engine = layerstream_engine_open(argv[1], "{\"ctx_size\": 512, \"use_gpu\": false}");
    if (!engine) {
        fprintf(stderr, "open failed: %s\n", layerstream_last_error());
        return 1;
    }

    LayerStreamCancelToken *tok = layerstream_cancel_token_new();
    char *summary = layerstream_engine_generate_stream(
        engine,
        "Hello",
        "{\"max_tokens\": 16, \"temperature\": 0.0}",
        tok,
        print_token,
        NULL
    );
    if (!summary) {
        fprintf(stderr, "\ngenerate failed: %s\n", layerstream_last_error());
        layerstream_cancel_token_free(tok);
        layerstream_engine_free(engine);
        return 1;
    }

    printf("\nSummary: %s\n", summary);
    layerstream_string_free(summary);
    layerstream_cancel_token_free(tok);
    layerstream_engine_free(engine);
    return 0;
}
