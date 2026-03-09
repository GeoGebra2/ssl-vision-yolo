#include "plugin_detect_yolo_candidates.h"
#include <cstdio>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <QByteArray>
#include <QBuffer>
#include <QImage>
#include <QDateTime>
#include <QString>
#include <QCoreApplication>
#include <opencv2/imgcodecs.hpp>
using namespace VarTypes;
 
 PluginDetectYoloCandidates::PluginDetectYoloCandidates(FrameBuffer* buffer)
     : VisionPlugin(buffer) {
   _settings = new VarList("YOLO Candidates");
   _settings->addChild(_enabled = new VarBool("Enable", false));
   _settings->addChild(_mock = new VarBool("Mock", true));
   _settings->addChild(_max_mock = new VarInt("Max Mock Detections", 0));
  _settings->addChild(_use_dnn = new VarBool("Use DNN", true));
  _settings->addChild(_model_path = new VarString("Model Path", "d:/ssl-vision-yolo/RoboCupVision/runs/detect/train/weights/best.onnx"));
  _settings->addChild(_input_w = new VarInt("Input Width", 640));
  _settings->addChild(_input_h = new VarInt("Input Height", 640));
  _settings->addChild(_swap_rb = new VarBool("Swap RB", false));
  _settings->addChild(_conf_th = new VarDouble("Conf Threshold", 0.25));
  _settings->addChild(_iou_th = new VarDouble("IoU Threshold", 0.5));
  _settings->addChild(_robot_class_ids = new VarString("Robot Class IDs (csv)", "1,2"));
  _settings->addChild(_ball_class_ids = new VarString("Ball Class IDs (csv)", "0"));
  _settings->addChild(_robot_blue_class_ids = new VarString("Robot Blue Class IDs (csv)", "1"));
  _settings->addChild(_robot_yellow_class_ids = new VarString("Robot Yellow Class IDs (csv)", "2"));
  _settings->addChild(_debug_print = new VarBool("Debug Print", false));
  _settings->addChild(_debug_net = new VarBool("Debug Net Verbose", false));
  _settings->addChild(_use_python = new VarBool("Use Python Backend", false));
  _settings->addChild(_py_command = new VarString("Python Command", "python"));
  _settings->addChild(_py_script = new VarString("Python Script", "d:/ssl-vision-yolo/RoboCupVision/yolo_stdio_service.py"));
  _settings->addChild(_py_args = new VarString("Python Args", ""));
  _settings->addChild(_py_timeout_ms = new VarInt("Python Timeout ms", 200));
  _settings->addChild(_py_use_jpeg = new VarBool("Send JPEG", true));
  _settings->addChild(_py_jpeg_quality = new VarInt("JPEG Quality", 80));
  _py = nullptr;
 }
 
 PluginDetectYoloCandidates::~PluginDetectYoloCandidates() {
   delete _settings;
   delete _enabled;
   delete _mock;
   delete _max_mock;
  delete _use_dnn;
  delete _model_path;
  delete _input_w;
  delete _input_h;
  delete _swap_rb;
  delete _conf_th;
  delete _iou_th;
  delete _robot_class_ids;
  delete _ball_class_ids;
  delete _robot_blue_class_ids;
  delete _robot_yellow_class_ids;
  delete _debug_print;
  delete _debug_net;
  delete _use_python;
  delete _py_command;
  delete _py_script;
  delete _py_args;
  delete _py_timeout_ms;
  delete _py_use_jpeg;
  delete _py_jpeg_quality;
  if (_py) { _py->stop(); delete _py; _py = nullptr; }
 }
 
 ProcessResult PluginDetectYoloCandidates::process(FrameData* data, RenderOptions* options) {
   (void)options;
   if (data == 0) return ProcessingFailed;
 
   Yolo::CandidateSet* cset = (Yolo::CandidateSet*)data->map.get("yolo_candidates");
   if (cset == 0) {
     cset = (Yolo::CandidateSet*)data->map.insert("yolo_candidates", new Yolo::CandidateSet());
   }
 
   cset->robots.clear();
  cset->robots_blue.clear();
  cset->robots_yellow.clear();
   cset->balls.clear();
 
   if (!_enabled->getBool()) {
     return ProcessingOk;
   }
 
#ifdef HAVE_OPENCV_DNN
  if (_use_python->getBool() && _use_dnn->getBool()) {
    _use_dnn->setBool(false);
  }
#endif
#ifdef HAVE_OPENCV_DNN
  if (_use_python->getBool()) {
    // ensure client exists and process started in this thread
    if (!_py) {
      _py = new PythonYoloClient();
    }
    if (!_py->isRunning()) {
      _py->start(_py_command->getString(), _py_script->getString(), _py_args->getString());
    }
    if (_py->isRunning()) {
      cv::Mat bgr;
      if (!toBGRMat(data->video, bgr)) {
        return ProcessingFailed;
      }
      std::string payload;
      if (_py_use_jpeg->getBool()) {
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, _py_jpeg_quality->getInt() };
        std::vector<uchar> buf;
        cv::imencode(".jpg", bgr, buf, params);
        QByteArray qbytes((const char*)buf.data(), (int)buf.size());
        QByteArray b64 = qbytes.toBase64();
        payload.assign(b64.constData(), (size_t)b64.size());
      } else {
        // raw BGR base64
        QByteArray qbytes((const char*)bgr.data, bgr.cols * bgr.rows * 3);
        QByteArray b64 = qbytes.toBase64();
        payload.assign(b64.constData(), (size_t)b64.size());
      }
      std::ostringstream oss;
      oss << "{";
      oss << "\"width\":" << bgr.cols << ",\"height\":" << bgr.rows;
      oss << ",\"format\":\"BGR\",\"jpeg\":" << (_py_use_jpeg->getBool() ? "true" : "false");
      oss << ",\"conf\":" << _conf_th->getDouble() << ",\"iou\":" << _iou_th->getDouble();
      oss << ",\"model\":\"" << _model_path->getString() << "\"";
      oss << ",\"image_b64\":\"" << payload << "\"";
      oss << "}";
      std::string reply;
      if (_py->requestLine(oss.str(), reply, _py_timeout_ms->getInt())) {
        // parse detections: support JSON array [{x1,...}] or CSV lines
        auto trim = [](std::string s)->std::string {
          size_t a = s.find_first_not_of(" \r\n\t");
          size_t b = s.find_last_not_of(" \r\n\t");
          if (a==std::string::npos) return std::string();
          return s.substr(a, b-a+1);
        };
        std::string r = trim(reply);
        if (!r.empty() && r[0] == '[') {
          // naive JSON array parse
          size_t pos = 0;
          while (true) {
            size_t p1 = r.find('{', pos);
            if (p1 == std::string::npos) break;
            size_t p2 = r.find('}', p1);
            if (p2 == std::string::npos) break;
            std::string obj = r.substr(p1+1, p2-p1-1);
            int x1=0,y1=0,x2=0,y2=0,cls=0; double conf=0.0;
            auto getVal = [&](const char* key)->std::string {
              std::string k = std::string(key);
              size_t kpos = obj.find(k);
              if (kpos==std::string::npos) return std::string();
              size_t colon = obj.find(':', kpos);
              if (colon==std::string::npos) return std::string();
              size_t comma = obj.find(',', colon+1);
              std::string val = obj.substr(colon+1, (comma==std::string::npos? obj.size(): comma) - (colon+1));
              return trim(val);
            };
            try {
              x1 = std::stoi(getVal("x1"));
              y1 = std::stoi(getVal("y1"));
              x2 = std::stoi(getVal("x2"));
              y2 = std::stoi(getVal("y2"));
              cls = std::stoi(getVal("class_id"));
              conf = std::stod(getVal("conf"));
              Yolo::Candidate cand;
              cand.x1 = x1; cand.y1 = y1; cand.x2 = x2; cand.y2 = y2; cand.conf = (float)conf; cand.class_id = cls;
              cset->robots.push_back(cand);
            } catch (...) {}
            pos = p2+1;
          }
        } else {
          // CSV per line: x1,y1,x2,y2,conf,class_id
          std::istringstream iss(r);
          std::string line;
          while (std::getline(iss, line)) {
            line = trim(line);
            if (line.empty()) continue;
            std::istringstream ls(line);
            std::string tok;
            std::vector<std::string> toks;
            while (std::getline(ls, tok, ',')) toks.push_back(trim(tok));
            if (toks.size() >= 6) {
              try {
                int x1 = std::stoi(toks[0]);
                int y1 = std::stoi(toks[1]);
                int x2 = std::stoi(toks[2]);
                int y2 = std::stoi(toks[3]);
                double conf = std::stod(toks[4]);
                int cls = std::stoi(toks[5]);
                Yolo::Candidate cand;
                cand.x1 = x1; cand.y1 = y1; cand.x2 = x2; cand.y2 = y2; cand.conf = (float)conf; cand.class_id = cls;
                cset->robots.push_back(cand);
              } catch (...) {}
            }
          }
        }
      } else {
        // timeout or failure
      }
      return ProcessingOk;
    }
  }
#endif

#ifdef HAVE_OPENCV_DNN
  if (_use_dnn->getBool() && !_use_python->getBool()) {
    const std::string modelPath = _model_path->getString();
    if (!_net_loaded || _loaded_model_path != modelPath) {
      try {
        _net = cv::dnn::readNetFromONNX(modelPath);
        _net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        _net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        _net_loaded = true;
        _loaded_model_path = modelPath;
        if (_debug_net && _debug_net->getBool()) {
          std::printf("YOLO: loaded ONNX '%s'\n", modelPath.c_str());
        }
      } catch (...) {
        _net_loaded = false;
        if (_debug_net && _debug_net->getBool()) {
          std::printf("YOLO: failed to load ONNX '%s'\n", modelPath.c_str());
        }
      }
    }
    if (_net_loaded) {
      cv::Mat bgr;
      if (!toBGRMat(data->video, bgr)) {
        return ProcessingFailed;
      }
      const int iw = _input_w->getInt();
      const int ih = _input_h->getInt();
      const int stride = 32;
      const int aiw = std::max(stride, (iw / stride) * stride);
      const int aih = std::max(stride, (ih / stride) * stride);
      float r = 1.0f;
      int dw = 0, dh = 0;
      cv::Mat letter;
      letterbox(bgr, letter, aiw, aih, r, dw, dh);
      if (_debug_net && _debug_net->getBool()) {
        if (aiw != iw || aih != ih) {
          std::printf("YOLO: input size aligned to stride32 (%d,%d) from (%d,%d)\n", aiw, aih, iw, ih);
        }
      }
      cv::Mat blob = cv::dnn::blobFromImage(letter, 1.0/255.0, cv::Size(aiw, aih), cv::Scalar(), _swap_rb->getBool(), false);
      if (_debug_net && _debug_net->getBool()) {
        std::printf("YOLO: input=%dx%d letter=%dx%d r=%.4f dw=%d dh=%d swapRB=%d\n",
                    bgr.cols, bgr.rows, letter.cols, letter.rows, r, dw, dh, _swap_rb->getBool()?1:0);
        int d0 = (blob.dims >= 1) ? blob.size[0] : 0;
        int d1 = (blob.dims >= 2) ? blob.size[1] : 0;
        int d2 = (blob.dims >= 3) ? blob.size[2] : 0;
        int d3 = (blob.dims >= 4) ? blob.size[3] : 0;
        std::printf("YOLO: blob dims=%d shape=(%d,%d,%d,%d)\n", blob.dims, d0, d1, d2, d3);
      }
      _net.setInput(blob);
      std::vector<cv::Mat> outs;
      try {
        _net.forward(outs, _net.getUnconnectedOutLayersNames());
      } catch (const cv::Exception& e) {
        if (_debug_net && _debug_net->getBool()) {
          std::printf("YOLO: forward failed: %s\n", e.what());
        }
        return ProcessingFailed;
      }
      if (_debug_net && _debug_net->getBool()) {
        std::printf("YOLO: outs=%zu\n", outs.size());
        for (size_t i = 0; i < outs.size(); ++i) {
          const cv::Mat& m = outs[i];
          if (m.dims == 2) {
            std::printf("outs[%zu]: dims=2 shape=(%d,%d)\n", i, m.rows, m.cols);
          } else if (m.dims == 3) {
            std::printf("outs[%zu]: dims=3 shape=(%d,%d,%d)\n", i, m.size[0], m.size[1], m.size[2]);
          } else if (m.dims == 4) {
            std::printf("outs[%zu]: dims=4 shape=(%d,%d,%d,%d)\n", i, m.size[0], m.size[1], m.size[2], m.size[3]);
          } else {
            std::printf("outs[%zu]: dims=%d\n", i, m.dims);
          }
        }
      }

      cv::Mat out;
      if (outs.size() == 1) {
        out = outs[0];
      } else if (!outs.empty()) {
        out = outs[0];
        for (size_t i = 1; i < outs.size(); ++i) {
          cv::hconcat(out, outs[i], out);
        }
      } else {
        return ProcessingOk;
      }
      if (_debug_net && _debug_net->getBool()) {
        if (out.dims == 2) {
          std::printf("YOLO: merged out dims=2 shape=(%d,%d)\n", out.rows, out.cols);
        } else if (out.dims == 3) {
          std::printf("YOLO: merged out dims=3 shape=(%d,%d,%d)\n", out.size[0], out.size[1], out.size[2]);
        } else if (out.dims == 4) {
          std::printf("YOLO: merged out dims=4 shape=(%d,%d,%d,%d)\n", out.size[0], out.size[1], out.size[2], out.size[3]);
        } else {
          std::printf("YOLO: merged out dims=%d\n", out.dims);
        }
      }

      cv::Mat det;
      if (out.dims == 3 && out.size[0] == 1) {
        int a = out.size[1];
        int b = out.size[2];
        if (b >= 5) {
          det = out.reshape(1, { a, b });
        } else if (a >= 5) {
          det = out.reshape(1, { b, a });
          det = det.t();
        } else {
          return ProcessingOk;
        }
      } else if (out.dims == 2) {
        int r = out.rows;
        int c = out.cols;
        if (c >= 5) {
          det = out;
        } else if (r >= 5) {
          det = out.t();
        } else {
          return ProcessingOk;
        }
      } else {
        return ProcessingOk;
      }

      const int num = det.rows;
      const int num_classes = det.cols - 4;
      if (num_classes <= 0) {
        return ProcessingOk;
      }
      if (_debug_net && _debug_net->getBool()) {
        std::printf("YOLO: det shape=(%d,%d) num_classes=%d\n", det.rows, det.cols, num_classes);
      }

      std::vector<int> cls_ids;
      std::vector<float> scores;
      std::vector<cv::Rect> boxes;
      boxes.reserve(num);
      scores.reserve(num);
      cls_ids.reserve(num);

      const float confTh = (float)_conf_th->getDouble();
      for (int i = 0; i < num; ++i) {
        const float* row = det.ptr<float>(i);
        float x = row[0];
        float y = row[1];
        float w = row[2];
        float h = row[3];
        int bestId = -1;
        float bestScore = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
          float sc = row[4 + c];
          if (sc > bestScore) { bestScore = sc; bestId = c; }
        }
        if (bestScore < confTh) continue;
        float x0 = (x - 0.5f*w - (float)dw) / r;
        float y0 = (y - 0.5f*h - (float)dh) / r;
        float x1 = (x + 0.5f*w - (float)dw) / r;
        float y1 = (y + 0.5f*h - (float)dh) / r;
        int xx = (int)std::max(0.0f, x0);
        int yy = (int)std::max(0.0f, y0);
        int ww = (int)std::min((float)bgr.cols-1, x1) - xx;
        int hh = (int)std::min((float)bgr.rows-1, y1) - yy;
        if (ww <= 0 || hh <= 0) continue;
        boxes.emplace_back(xx, yy, ww, hh);
        scores.emplace_back(bestScore);
        cls_ids.emplace_back(bestId);
      }

      std::vector<int> keep;
      cv::dnn::NMSBoxes(boxes, scores, confTh, (float)_iou_th->getDouble(), keep);
      if (_debug_net && _debug_net->getBool()) {
        std::printf("YOLO: NMS kept=%zu\n", keep.size());
      }

      std::vector<int> robot_ids, ball_ids, blue_ids, yellow_ids;
      parseClassIdList(_robot_class_ids->getString(), robot_ids);
      parseClassIdList(_ball_class_ids->getString(), ball_ids);
      parseClassIdList(_robot_blue_class_ids->getString(), blue_ids);
      parseClassIdList(_robot_yellow_class_ids->getString(), yellow_ids);

      auto isIn = [](int id, const std::vector<int>& arr)->bool {
        for (int v : arr) if (v == id) return true;
        return false;
      };

      for (int idx : keep) {
        const cv::Rect& b = boxes[(size_t)idx];
        Yolo::Candidate cand;
        cand.x1 = b.x;
        cand.y1 = b.y;
        cand.x2 = b.x + b.width;
        cand.y2 = b.y + b.height;
        cand.conf = scores[(size_t)idx];
        cand.class_id = cls_ids[(size_t)idx];
        bool is_robot = isIn(cand.class_id, robot_ids) || isIn(cand.class_id, blue_ids) || isIn(cand.class_id, yellow_ids);
        if (is_robot) cset->robots.push_back(cand);
        if (isIn(cand.class_id, blue_ids)) cset->robots_blue.push_back(cand);
        if (isIn(cand.class_id, yellow_ids)) cset->robots_yellow.push_back(cand);
        if (isIn(cand.class_id, ball_ids)) {
          cset->balls.push_back(cand);
        }
      }

      if (_debug_print && _debug_print->getBool()) {
        std::printf("YOLO: balls=%zu blue=%zu yellow=%zu\n",
                    cset->balls.size(), cset->robots_blue.size(), cset->robots_yellow.size());
        for (size_t i = 0; i < cset->balls.size(); ++i) {
          const auto &c = cset->balls[i];
          std::printf("ball #%zu: [%d,%d,%d,%d] conf=%.3f class=%d\n",
                      i, c.x1, c.y1, c.x2, c.y2, c.conf, c.class_id);
        }
        for (size_t i = 0; i < cset->robots_blue.size(); ++i) {
          const auto &c = cset->robots_blue[i];
          std::printf("robot_blue #%zu: [%d,%d,%d,%d] conf=%.3f class=%d\n",
                      i, c.x1, c.y1, c.x2, c.y2, c.conf, c.class_id);
        }
        for (size_t i = 0; i < cset->robots_yellow.size(); ++i) {
          const auto &c = cset->robots_yellow[i];
          std::printf("robot_yellow #%zu: [%d,%d,%d,%d] conf=%.3f class=%d\n",
                      i, c.x1, c.y1, c.x2, c.y2, c.conf, c.class_id);
        }
      }
      return ProcessingOk;
    }
  }
#endif
   if (_mock->getBool()) {
     int w = data->video.getWidth();
     int h = data->video.getHeight();
     int n = _max_mock->getInt();
     if (n < 0) n = 0;
     for (int i = 0; i < n; i++) {
       Yolo::Candidate r;
       int cx = (i + 1) * w / (n + 1);
       int cy = (i + 1) * h / (n + 1);
       int rw = w / 10;
       int rh = h / 10;
       r.x1 = cx - rw / 2;
       r.y1 = cy - rh / 2;
       r.x2 = cx + rw / 2;
       r.y2 = cy + rh / 2;
       r.conf = 0.5f;
       r.class_id = 0;
      cset->robots.push_back(r);
      if ((i % 2) == 0) {
        cset->robots_blue.push_back(r);
      } else {
        cset->robots_yellow.push_back(r);
      }
     }
   }
  if (_debug_print && _debug_print->getBool()) {
    std::printf("YOLO: balls=%zu blue=%zu yellow=%zu\n",
                cset->balls.size(), cset->robots_blue.size(), cset->robots_yellow.size());
    for (size_t i = 0; i < cset->balls.size(); ++i) {
      const auto &c = cset->balls[i];
      std::printf("ball #%zu: [%d,%d,%d,%d] conf=%.3f class=%d\n",
                  i, c.x1, c.y1, c.x2, c.y2, c.conf, c.class_id);
    }
    for (size_t i = 0; i < cset->robots_blue.size(); ++i) {
      const auto &c = cset->robots_blue[i];
      std::printf("robot_blue #%zu: [%d,%d,%d,%d] conf=%.3f class=%d\n",
                  i, c.x1, c.y1, c.x2, c.y2, c.conf, c.class_id);
    }
    for (size_t i = 0; i < cset->robots_yellow.size(); ++i) {
      const auto &c = cset->robots_yellow[i];
      std::printf("robot_yellow #%zu: [%d,%d,%d,%d] conf=%.3f class=%d\n",
                  i, c.x1, c.y1, c.x2, c.y2, c.conf, c.class_id);
    }
  }
 
   return ProcessingOk;
 }
 
 VarList* PluginDetectYoloCandidates::getSettings() { return _settings; }
 
 std::string PluginDetectYoloCandidates::getName() { return "YoloCandidates"; }

#ifdef HAVE_OPENCV_DNN
void PluginDetectYoloCandidates::letterbox(const cv::Mat& src, cv::Mat& dst, int new_w, int new_h, float& r, int& dw, int& dh) {
  int width = src.cols;
  int height = src.rows;
  r = std::min((float)new_w / (float)width, (float)new_h / (float)height);
  int w = (int)std::round(width * r);
  int h = (int)std::round(height * r);
  cv::Mat resized;
  cv::resize(src, resized, cv::Size(w, h));
  dw = (new_w - w) / 2;
  dh = (new_h - h) / 2;
  cv::copyMakeBorder(resized, dst, dh, new_h - h - dh, dw, new_w - w - dw, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
}

void PluginDetectYoloCandidates::parseClassIdList(const std::string& s, std::vector<int>& out) {
  out.clear();
  int val = 0;
  bool in_num = false;
  bool neg = false;
  for (size_t i = 0; i <= s.size(); ++i) {
    char c = (i < s.size() ? s[i] : ',');
    if (c == '-' && !in_num) { neg = true; continue; }
    if (c >= '0' && c <= '9') { val = val * 10 + (c - '0'); in_num = true; continue; }
    if (c == ',' || i == s.size()) {
      if (in_num) {
        out.push_back(neg ? -val : val);
        val = 0; in_num = false; neg = false;
      }
    }
  }
}

bool PluginDetectYoloCandidates::toBGRMat(const RawImage& src, cv::Mat& bgr) {
  const int w = src.getWidth();
  const int h = src.getHeight();
  if (w <= 0 || h <= 0) return false;
  switch (src.getColorFormat()) {
    case COLOR_RGB8: {
      cv::Mat rgb(h, w, CV_8UC3, (void*)src.getData());
      cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
      return true;
    }
    case COLOR_YUV422_UYVY: {
      cv::Mat uyvy(h, w, CV_8UC2, (void*)src.getData());
      cv::cvtColor(uyvy, bgr, cv::COLOR_YUV2BGR_UYVY);
      return true;
    }
    case COLOR_YUV444: {
      cv::Mat yuv(h, w, CV_8UC3, (void*)src.getData());
      cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR);
      return true;
    }
    default: {
      bgr = cv::Mat(h, w, CV_8UC3);
      for (int y = 0; y < h; ++y) {
        cv::Vec3b* row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; ++x) {
          rgb c = src.getRgb(x, y);
          row[x] = cv::Vec3b(c.b, c.g, c.r);
        }
      }
      return true;
    }
  }
}
#endif
