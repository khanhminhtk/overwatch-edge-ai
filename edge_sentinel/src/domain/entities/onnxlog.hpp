#pragma once
#include <string>

struct LogConfig {
    std::string deviceID;
    bool isDebugMode;
    
    LogConfig(const std::string& id, bool debug)
        : deviceID(id), isDebugMode(debug) {}
};