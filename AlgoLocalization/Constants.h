#pragma once

// Hyperparameters for tuning
#define RADIUS 50.0
// L2 + CHI SQUARES
//#define RADIUS_SIFT 80.0
//#define RADIUS_COLOR 16
//#define FRONTIER_SIFT 6000
//#define FRONTIER_COLOR 500

// KL-DIVERGENCE

#define RADIUS_SIFT 200
#define RADIUS_COLOR 30 // Non-normalized
#define ENABLE_HISTOGRAM_NORMALIZATION true
#define RADIUS_COLOR 0.1
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


#define SALON 0
#define CUISINE 1
#define REUNION 2

#define WORD_TYPES 7
#define VERBOSE false

#define ENABLE_CORRECTION false