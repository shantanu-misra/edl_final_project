/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * ...
 */

/* Includes ---------------------------------------------------------------- */
#include <Arduino.h>              // for analogRead(), millis(), delay()
#include <EDL-2_v1_inferencing.h> // your updated 8kHz inference headers

/*** Improved analog mic sampling for 8kHz ***/
#define INPUT_PIN       A0
#define SAMPLE_RATE     8000      // 8kHz sample rate
#define SAMPLE_INTERVAL 125       // Microseconds between samples (1000000/8000)

/** Fusion settings **/
#define FUSION_WINDOW 5

// circular buffer to hold last 5 inference probability vectors
static float classification_buffer[FUSION_WINDOW][EI_CLASSIFIER_LABEL_COUNT] = {0};
static int fusion_index = 0;
static bool fusion_buffer_full = false;

// recency weights (sum to 1)
const float fusion_weights[FUSION_WINDOW] = { 0.05f, 0.10f, 0.20f, 0.30f, 0.35f };
// boost factors for [Engine, Knock, Neither]
const float class_boost[EI_CLASSIFIER_LABEL_COUNT] = { 1.0f, 2.0f, 1.0f };
// threshold for declaring a knock
const float KNOCK_THRESHOLD = 0.25f;

/** Audio buffers, pointers and selectors **/
typedef struct {
    int16_t *buffer;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static bool debug_nn = false;

/**
 * @brief      Initialize inferencing struct
 */
static bool microphone_inference_start(uint32_t n_samples) {
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));
    if (inference.buffer == NULL) {
        return false;
    }
    inference.n_samples = n_samples;
    inference.buf_count = 0;
    return true;
}

/**
 * @brief      Improved recording function with timeout and fixed timing
 */
static bool microphone_inference_record(void) {
    inference.buf_count = 0;
    unsigned long start_time = millis();
    unsigned long next_sample_time = micros();
    unsigned long last_report_time = start_time;

    // Maximum recording time (10 seconds timeout)
    const unsigned long timeout_ms = 10000;

    while (inference.buf_count < inference.n_samples) {
        // Check for timeout
        if (millis() - start_time > timeout_ms) {
            ei_printf("Recording timeout! Collected %lu/%lu samples\n", 
                      inference.buf_count, inference.n_samples);
            return false;
        }

        // Time to take a new sample?
        if (micros() >= next_sample_time) {
            inference.buffer[inference.buf_count++] = (int16_t)analogRead(INPUT_PIN);
            next_sample_time += SAMPLE_INTERVAL;

            // Report progress every second
            unsigned long current_time = millis();
            if (current_time - last_report_time > 1000) {
                ei_printf("Recording progress: %lu/%lu samples (%d%%)\n", 
                          inference.buf_count, inference.n_samples,
                          (int)(inference.buf_count * 100 / inference.n_samples));
                last_report_time = current_time;
            }
        }
        delayMicroseconds(10);
    }

    ei_printf("Recording completed in %lu ms\n", millis() - start_time);
    return true;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset,
                                            size_t length,
                                            float *out_ptr) {
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
    return 0;
}

/**
 * @brief      Stop and free buffers
 */
static void microphone_inference_end(void) {
    free(inference.buffer);
}

/**
 * @brief      Arduino setup function
 */
void setup() {
    Serial.begin(115200);
    delay(1000); // Give serial time to initialize
    Serial.println("Edge Impulse Inferencing Demo");

    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n",
              EI_CLASSIFIER_RAW_SAMPLE_COUNT / (SAMPLE_RATE / 1000));
    ei_printf("\tNo. of classes: %d\n",
              sizeof(ei_classifier_inferencing_categories)
              / sizeof(ei_classifier_inferencing_categories[0]));
    ei_printf("\tActual sample rate: %d Hz\n", SAMPLE_RATE);

    #ifdef ARDUINO_ARCH_SAMD
    analogReadResolution(12); // 12-bit resolution for SAMD boards
    #endif

    if (!microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT)) return;
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop() {
    ei_printf("Starting inferencing in 2 seconds...\n");
    delay(2000);

    ei_printf("Recording...\n");
    if (!microphone_inference_record()) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }
    ei_printf("Recording done\n");

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = { 0 };

    EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        microphone_inference_end();
        return;
    }

    // --- FUSION: push this inference into the circular buffer ---
    for (size_t c = 0; c < EI_CLASSIFIER_LABEL_COUNT; c++) {
        classification_buffer[fusion_index][c] = result.classification[c].value;
    }
    fusion_index++;
    if (fusion_index >= FUSION_WINDOW) {
        fusion_index = 0;
        fusion_buffer_full = true;
    }

    // --- compute & print weighted, boosted average once buffer is full ---
    if (fusion_buffer_full) {
        float S[EI_CLASSIFIER_LABEL_COUNT] = {0};

        // weighted sum
        for (int i = 0; i < FUSION_WINDOW; i++) {
            for (size_t c = 0; c < EI_CLASSIFIER_LABEL_COUNT; c++) {
                S[c] += fusion_weights[i] * classification_buffer[i][c];
            }
        }
        // apply class boost
        for (size_t c = 0; c < EI_CLASSIFIER_LABEL_COUNT; c++) {
            S[c] *= class_boost[c];
        }

        // check knock threshold
        // (knock is index 1 in categories array)
        if (S[1] > KNOCK_THRESHOLD) {
            Serial.println("KNOCK DETECTED!");
        }
        else {
            Serial.println("NO KNOCK DETECTED");
        }

        // print only the fused values
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            const char* label = result.classification[ix].label;
            Serial.print(label);
            Serial.print(": ");
            Serial.println(S[ix], 5);
        }
    }

    delay(1000);
}
