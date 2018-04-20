#pragma once

// Hyperparameters for tuning
#define RADIUS 50.0
// L2 + CHI SQUARES
//#define RADIUS_SIFT 80.0
//#define RADIUS_COLOR 16
//#define FRONTIER_SIFT 6000
//#define FRONTIER_COLOR 500


// RADIUS_SIFT for different norms: L2 ~ 200, CHI_Squares ~ 700, 
// RADIUS_COLOR for non-normalized: 30 ~ 50, normalized Diffusion ~ 0.03


#define RADIUS_SIFT 200
#define ENABLE_HISTOGRAM_NORMALIZATION false
#define RADIUS_COLOR 30
//#define RADIUS_COLOR 0.025

#define NUM_MAX_SIFT 250
#define FRONTIER_SIFT 8000
#define FRONTIER_COLOR 8000
#define NUM_MAX_WORDS 500
#define K_SPLIT 10
#define MAX_CHILD_NUM 3
#define FULL_SEARCH true
#define ENABLE_EQUALIZER false

// Really constant 
#define NUM_ROOMS 3
#define THRESHOLD_FIRST_VOTE 0.1
#define THRESHOLD_SECOND_VOTE 0.3
#define DIM_COLOR_HIST 16
#define DIM_SIFT 128
#define FEATURE_SIFT 1
#define FEATURE_COLOR 0
#define VIDEO_SIZE 50
#define TRAIN_SIZE 25

#define SALON 0
#define CUISINE 1
#define REUNION 2

#define WORD_TYPES 7
#define VERBOSE false

#define ENABLE_CORRECTION false

#define NORM_KL 0
#define NORM_DIFFUSION 1
const int mNorm = NORM_KL;
#define DISP_DEBUG false

#define N_LEARNING TRAIN_SIZE
#define N_TEST 25
#define N_IMGS 2
#define N_EXPERIMENTS 3
