#include <iostream>
#include <nlohmann/json.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(LogTest, Basic) {
    LOG(INFO) << "This is an info log.";
    LOG(WARNING) << "This is a warning log.";
    LOG(ERROR) << "This is an error log.";
}

TEST(JSONTest, Parse) {
    std::string json_str = R"({"name": "John", "age": 30, "city": "New York"})";
    auto json_obj = nlohmann::json::parse(json_str);
    
    EXPECT_EQ(json_obj["name"], "John");
    EXPECT_EQ(json_obj["age"], 30);
    EXPECT_EQ(json_obj["city"], "New York");
}