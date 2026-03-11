 #pragma once
 #include "visionplugin.h"
 #include "yolo_candidates.h"
 #include "VarTypes.h"
#include "python_yolo_client.h"
 
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>

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
  VarInt* _py_submit_every_n;
  VarInt* _py_max_side;

  cv::dnn::Net _net;
  bool _net_loaded = false;
  std::string _loaded_model_path;
  static void letterbox(const cv::Mat& src, cv::Mat& dst, int new_w, int new_h, float& r, int& dw, int& dh);
  static void parseClassIdList(const std::string& s, std::vector<int>& out);
  static bool toBGRMat(const RawImage& src, cv::Mat& bgr);
  static bool parsePythonReply(const std::string& reply, std::vector<Yolo::Candidate>& out);
  static void fillCandidateSet(const std::vector<Yolo::Candidate>& src, Yolo::CandidateSet* cset,
                               const std::vector<int>& robot_ids, const std::vector<int>& ball_ids,
                               const std::vector<int>& blue_ids, const std::vector<int>& yellow_ids);
  struct PythonTask {
    cv::Mat bgr;
    std::string command;
    std::string script;
    std::string args;
    std::string model_path;
    double conf = 0.25;
    double iou = 0.5;
    int timeout_ms = 200;
    bool use_jpeg = true;
    int jpeg_quality = 80;
    double scale_x = 1.0;
    double scale_y = 1.0;
  };
  std::thread _py_worker;
  std::mutex _py_mutex;
  std::condition_variable _py_cv;
  bool _py_worker_started = false;
  bool _py_stop = false;
  bool _py_has_pending = false;
  PythonTask _py_pending_task;
  std::vector<Yolo::Candidate> _py_last_raw;
  bool _py_last_ready = false;
  int _py_drop_count = 0;
  int _py_frame_index = 0;
  void ensurePythonWorker();
  void stopPythonWorker();
  void pythonWorkerMain();
 public:
   PluginDetectYoloCandidates(FrameBuffer* buffer);
   ~PluginDetectYoloCandidates() override;
   ProcessResult process(FrameData* data, RenderOptions* options) override;
   VarList* getSettings() override;
   std::string getName() override;
 };
