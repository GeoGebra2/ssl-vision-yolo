 #pragma once
 #include "visionplugin.h"
 #include "yolo_candidates.h"
 #include "VarTypes.h"
#include "python_yolo_client.h"
 
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

 class PluginDetectYoloCandidates : public VisionPlugin {
 protected:
   VarList* _settings;
   VarBool* _enabled;
   VarBool* _mock;
   VarInt* _max_mock;
  VarBool* _use_dnn;
  VarString* _model_path;
  VarInt* _input_w;
  VarInt* _input_h;
  VarBool* _swap_rb;
  VarDouble* _conf_th;
  VarDouble* _iou_th;
  VarString* _robot_class_ids;
  VarString* _ball_class_ids;
  VarString* _robot_blue_class_ids;
  VarString* _robot_yellow_class_ids;
  VarBool* _debug_print;
  VarBool* _debug_net;
  VarBool* _use_python;
  VarString* _py_command;
  VarString* _py_script;
  VarString* _py_args;
  VarInt* _py_timeout_ms;
  VarBool* _py_use_jpeg;
  VarInt* _py_jpeg_quality;

  cv::dnn::Net _net;
  bool _net_loaded = false;
  std::string _loaded_model_path;
  static void letterbox(const cv::Mat& src, cv::Mat& dst, int new_w, int new_h, float& r, int& dw, int& dh);
  static void parseClassIdList(const std::string& s, std::vector<int>& out);
  static bool toBGRMat(const RawImage& src, cv::Mat& bgr);
  PythonYoloClient* _py;
 public:
   PluginDetectYoloCandidates(FrameBuffer* buffer);
   ~PluginDetectYoloCandidates() override;
   ProcessResult process(FrameData* data, RenderOptions* options) override;
   VarList* getSettings() override;
   std::string getName() override;
 };
