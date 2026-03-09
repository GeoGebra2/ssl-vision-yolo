#pragma once
#include <string>
#include <QProcess>
#include <QByteArray>
#include <QString>
#include <QMutex>
#include <QWaitCondition>
class PythonYoloClient {
public:
  PythonYoloClient();
  ~PythonYoloClient();
  bool start(const std::string& command, const std::string& script, const std::string& args);
  bool isRunning() const;
  void stop();
  bool requestLine(const std::string& line, std::string& out, int timeout_ms);
private:
  QProcess _proc;
  QMutex _mutex;
};
