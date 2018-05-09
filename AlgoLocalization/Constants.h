#pragma once

using namespace std;
// ---------------- Dictionary parameters ----------------
#define RADIUS 50.0
#define RADIUS_SIFT 180 // SIFT
//#define RADIUS_SIFT 120 // ORB
//#define RADIUS_SIFT 0.20 // SURF
#define ENABLE_HISTOGRAM_NORMALIZATION true
#define RADIUS_COLOR 0.035
//#define RADIUS_COLOR 0.025

#define NUM_MAX_SIFT 180
#define FRONTIER_SIFT 8000
#define FRONTIER_COLOR 8000
#define NUM_MAX_WORDS 250
#define K_SPLIT 5
#define MAX_CHILD_NUM 1

// ----------------Really constant ----------------
#define THRESHOLD_FIRST_VOTE 0.1
#define THRESHOLD_SECOND_VOTE 0.25
#define DIM_COLOR_HIST 16
#define DIM_SIFT 128
#define USE_SIFT 1
#define USE_COLOR 0
#define USE_SURF 2
#define WORD_TYPES 7
#define NUM_MIN_FEATURES 3000
#define NUM_MAX_IMAGES 5


// ------------ Model ----------------
#define FULL_SEARCH true
#define ENABLE_CLAHE false
#define ENABLE_CORRECTION false
#define ENABLE_INCREMENTAL true
#define PRIORITIZE_SIFT true
#define NORM_KL 0
#define NORM_DIFFUSION 1
const int mNorm = NORM_KL;
#define ENABLE_EQUALIZER false

// ------------ Display --------------
#define DISP_DEBUG false
#define DISP_IMAGE false
#define DISP_INCREMENTAL false
#define VERBOSE true

// ------------ Hyperparameters ------------
#define TEST_SIZE 30
#define TRAIN_SIZE 50
#define N_LEARNING TRAIN_SIZE
#define N_TEST 50
#define N_IMGS 1
#define N_EXPERIMENTS 1
#define WEIGHT_COLOR 0.60 // 0.5


// Datasets 
string salonTrainPath;
string cuisineTrainPath;
string reunionTrainPath;
string mangerTrainPath;
string salonTestPath;
string cuisineTestPath;
string reunionTestPath;
string mangerTestPath;
#define FOLDER_V2 "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\"
#define SALON_TRAIN_V2 "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\salon_train\\"
#define CUISINE_TRAIN_V2 "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\cuisine_train\\"
#define REUNION_TRAIN_V2 "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\reunion_train\\"
#define MANGER_TRAIN_V2 "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\manger_train\\"

#define SALON_TEST_V2 "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\salon_test\\"
#define CUISINE_TEST_V2 "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\cuisine_test\\"
#define REUNION_TEST_V2 "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\reunion_test\\"
#define MANGER_TEST_V2 "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\manger_test\\"

#define CUISINE_TEST_V2_VAR "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\cuisine_test_var\\"
#define SALON_TEST_V2_VAR "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\salon_test_var\\"
#define REUNION_TEST_V2_VAR "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\reunion_test_var\\"
#define MANGER_TEST_V2_VAR "D:\\WorkSpace\\03_Resources\\Dataset\\v2\\manger_test_var\\"

#define DATA_V1 1
#define DATA_V2 2
#define DATA_V2_VAR 3
const int dataSet = DATA_V2_VAR;

void initParameters()
{
    if (dataSet == DATA_V2)
    {
        salonTrainPath = SALON_TRAIN_V2;
        cuisineTrainPath = CUISINE_TRAIN_V2;
        reunionTrainPath = REUNION_TRAIN_V2;
        salonTestPath = SALON_TEST_V2;
        cuisineTestPath = CUISINE_TEST_V2;
        reunionTestPath = REUNION_TEST_V2;
        mangerTrainPath = MANGER_TRAIN_V2;
        mangerTestPath = MANGER_TRAIN_V2;
    }

    if (dataSet == DATA_V2_VAR)
    {
        salonTrainPath = SALON_TRAIN_V2;
        cuisineTrainPath = CUISINE_TRAIN_V2;
        reunionTrainPath = REUNION_TRAIN_V2;
        mangerTrainPath = MANGER_TRAIN_V2;
        salonTestPath = SALON_TEST_V2_VAR;
        cuisineTestPath = CUISINE_TEST_V2_VAR;
        reunionTestPath = REUNION_TEST_V2_VAR;
        mangerTestPath = MANGER_TEST_V2_VAR;
    }
}


// Rooms
struct Config
{
    int NUM_ROOMS = 2;

    template <class Archive>
    void serialize(Archive& ar)
    {
        ar(NUM_ROOMS);
    }
};

Config mConfig;
void SetNumRoom(int numRoom)
{
    mConfig.NUM_ROOMS = numRoom;
}

void AddNumRoom()
{
    mConfig.NUM_ROOMS++;
}

int GetNumRoom()
{
    return mConfig.NUM_ROOMS;
}

#define SALON 0
#define CUISINE 1
#define REUNION 2
#define MANGER 3


