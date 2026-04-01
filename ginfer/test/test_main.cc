#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cstdlib>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);

  FLAGS_logtostderr = 1;
  if (const char* v = std::getenv("GLOG_stderrthreshold")) {
    FLAGS_stderrthreshold = std::atoi(v);
  }
  if (const char* v = std::getenv("GLOG_minloglevel")) {
    FLAGS_minloglevel = std::atoi(v);
  }

  LOG(INFO) << "Starting tests...";
  return RUN_ALL_TESTS();
}