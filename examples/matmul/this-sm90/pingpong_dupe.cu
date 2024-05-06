#include "common.h"
#include "reference.h"

const int testM = 5120;
const int testN = 4096;
const int testK = 2048;

static constexpr int cluster_M = 2;
static constexpr int cluster_N = 1;
static constexpr int wg_number = 3;

static constexpr int blockM = 128;
static constexpr int blockN = 128;
static constexpr int blockK = 64;
static constexpr int stages = 7;
