#include "python_yolo_client.h"
#include <QMutexLocker>
#include <QStringList>

PythonYoloClient::PythonYoloClient() {}
PythonYoloClient::~PythonYoloClient() { stop(); }

bool PythonYoloClient::start(const std::string& command, const std::string& script, const std::string& args) {
  QMutexLocker locker(&_mutex);
  if (_proc.state() != QProcess::NotRunning) return true;
  QString cmd = QString::fromStdString(command);
  QStringList arglist;
  if (!script.empty()) arglist << QString::fromStdString(script);
  if (!args.empty()) {
    // naive split on spaces; users can pass quoted args if needed
    arglist << QString::fromStdString(args).split(' ', Qt::SkipEmptyParts);
  }
  _proc.setProcessChannelMode(QProcess::SeparateChannels);
  _proc.start(cmd, arglist, QProcess::ReadWrite);
  bool ok = _proc.waitForStarted(3000);
  return ok;
}

bool PythonYoloClient::isRunning() const {
  return _proc.state() != QProcess::NotRunning;
}

void PythonYoloClient::stop() {
  QMutexLocker locker(&_mutex);
  if (_proc.state() != QProcess::NotRunning) {
    _proc.closeWriteChannel();
    _proc.terminate();
    _proc.waitForFinished(1000);
    if (_proc.state() != QProcess::NotRunning) {
      _proc.kill();
      _proc.waitForFinished(1000);
    }
  }
}

bool PythonYoloClient::requestLine(const std::string& line, std::string& out, int timeout_ms) {
  QMutexLocker locker(&_mutex);
  if (_proc.state() == QProcess::NotRunning) return false;
  QByteArray in = QByteArray::fromStdString(line);
  in.append('\n');
  _proc.write(in);
  // QProcess has no flush(); ensure data is written
  _proc.waitForBytesWritten(timeout_ms);
  if (!_proc.waitForReadyRead(timeout_ms)) return false;
  QByteArray all;
  // read one line
  all = _proc.readLine();
  if (all.isEmpty()) {
    // try readAll if no line
    all = _proc.readAllStandardOutput();
  }
  out.assign(all.constData(), (size_t)all.size());
  return !out.empty();
}
