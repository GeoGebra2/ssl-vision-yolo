//========================================================================
//  This software is free: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License Version 3,
//  as published by the Free Software Foundation.
//
//  This software is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  Version 3 in the file COPYING that came with this distribution.
//  If not, see <http://www.gnu.org/licenses/>.
//========================================================================
/*!
  \file    cmpattern_teamdetector.cpp
  \brief   C++ Implementation: teamdetector
  \author  Original CMDragons Robot Detection Code (C), James Bruce
           ssl-vision restructuring and modifications, Stefan Zickler, 2009
*/
//========================================================================
#include "cmpattern_teamdetector.h"

namespace CMPattern {

TeamDetectorSettings::TeamDetectorSettings(string external_file) {
 settings=new VarList("Robot Detection");
 settings->addChild(robotPatternSettings = new VarList("Pattern"));
 robotPattern = new RobotPattern(robotPatternSettings);
 if (external_file.length()==0) {
    settings->addChild(teams = new VarList("Teams"));; // a global variable, defining all teams
 } else {
    settings->addChild(teams = new VarExternal(external_file,"Teams"));; // a global variable, defining all teams
 }
 connect(teams,SIGNAL(childAdded(VarType *)),this,SLOT(slotTeamNodeAdded(VarType *)));
 settings->addChild(addTeam = new VarTrigger("Add", "Add Team..."));
 connect(addTeam,SIGNAL(signalTriggered()),this,SLOT(slotAddPressed()));

  connect(robotPattern,SIGNAL(signalChangeOccured(VarType*)),this,SLOT(slotTeamDataChanged()));
}

void TeamDetectorSettings::slotTeamNodeAdded(VarType * node) {
  team_vector.push_back(new Team((VarList *)node));
  connect(team_vector[team_vector.size()-1],SIGNAL(signalTeamNameChanged()),this,SIGNAL(teamInfoChanged()));
  emit(teamInfoChanged());
}

void TeamDetectorSettings::slotAddPressed() {
  teams->addChild(new VarList("New Team " + QString::number(teams->getChildrenCount()).toStdString()));
}

vector<Team *> TeamDetectorSettings::getTeams() const {
  return team_vector;
}

Team * TeamDetectorSettings::getTeam(int idx) const {
  if (idx < 0 || idx >= (int)team_vector.size()) return 0;
  return (team_vector[idx]);
}

TeamDetector::TeamDetector(LUT3D * lut3d, const CameraParameters& camera_params, const RoboCupField& field) : _camera_params(camera_params), _field(field) {
  _robotPattern=0;
  _lut3d=lut3d;

  histogram=0;

  color_id_yellow = _lut3d->getChannelID("Yellow");
  if (color_id_yellow == -1) printf("WARNING color label 'Yellow' not defined in LUT!!!\n");
  color_id_blue = _lut3d->getChannelID("Blue");
  if (color_id_blue == -1) printf("WARNING color label 'Blue' not defined in LUT!!!\n");  

  color_id_cyan = _lut3d->getChannelID("Cyan");
  if (color_id_cyan == -1) printf("WARNING color label 'Cyan' not defined in LUT!!!\n");

  color_id_pink = _lut3d->getChannelID("Pink");
  if (color_id_pink == -1) printf("WARNING color label 'Pink' not defined in LUT!!!\n");

  color_id_green = _lut3d->getChannelID("Green");
  if (color_id_green == -1) printf("WARNING color label 'Green' not defined in LUT!!!\n");

  color_id_black = _lut3d->getChannelID("Black");
  if (color_id_black == -1) printf("WARNING color label 'Black' not defined in LUT!!!\n");

  color_id_white = _lut3d->getChannelID("White");
  if (color_id_white == -1) printf("WARNING color label 'White' not defined in LUT!!!\n");  

  color_id_clear = 0;

  color_id_green = _lut3d->getChannelID("Green");
  if (color_id_green == -1) printf("WARNING color label 'Green' not defined in LUT!!!\n");  

  color_id_field_green = _lut3d->getChannelID("Field Green");
  if (color_id_field_green == -1) printf("WARNING color label 'Field Green' not defined in LUT!!!\n");
}

void TeamDetector::init(RobotPattern * robotPattern, Team * team) {
  _robotPattern=robotPattern;
  _team=team;

  if (histogram==0) histogram= new CMVision::Histogram(_lut3d->getChannelCount());

  //--------------THINGS THAT MIGHT CHANGE DURING RUNTIME BELOW:------------
  //update field:
  field_filter.update(_field);

  //read config:
  _unique_patterns=_robotPattern->_unique_patterns->getBool();
  _have_angle=_robotPattern->_have_angle->getBool();
  _load_markers_from_image_file=_robotPattern->_load_markers_from_image_file->getBool();
  _marker_image_file=_robotPattern->_marker_image_file->getString();
  _marker_image_rows=_robotPattern->_marker_image_rows->getInt();
  _marker_image_cols=_robotPattern->_marker_image_cols->getInt();
  _robot_height=_team->_robot_height->getDouble();

  _center_marker_area_mean=_robotPattern->_center_marker_area_mean->getDouble();
  _center_marker_area_stddev=_robotPattern->_center_marker_area_stddev->getDouble();
  _center_marker_uniform=_robotPattern->_center_marker_uniform->getDouble();
  _center_marker_duplicate_distance=_robotPattern->_center_marker_duplicate_distance->getDouble();

  _other_markers_max_detections=_robotPattern->_other_markers_max_detections->getInt();
  _other_markers_max_query_distance=_robotPattern->_other_markers_max_query_distance->getDouble();

  filter_team.setWidth(_robotPattern->_center_marker_min_width->getInt(),robotPattern->_center_marker_max_width->getInt());
  filter_team.setHeight(_robotPattern->_center_marker_min_height->getInt(),robotPattern->_center_marker_max_height->getInt());
  filter_team.setArea(_robotPattern->_center_marker_min_area->getInt(),robotPattern->_center_marker_max_area->getInt());

  filter_others.setWidth(_robotPattern->_other_markers_min_width->getInt(),robotPattern->_other_markers_max_width->getInt());
  filter_others.setHeight(_robotPattern->_other_markers_min_height->getInt(),robotPattern->_other_markers_max_height->getInt());
  filter_others.setArea(_robotPattern->_other_markers_min_area->getInt(),robotPattern->_other_markers_max_area->getInt());

  _histogram_enable=_robotPattern->_histogram_enable->getBool();
  _histogram_pixel_scan_radius=_robotPattern->_histogram_pixel_scan_radius->getInt();

  _histogram_markeryness.set(_robotPattern->_histogram_min_markeryness->getDouble(),_robotPattern->_histogram_max_markeryness->getDouble());
  _histogram_field_greenness.set(_robotPattern->_histogram_min_field_greenness->getDouble(),_robotPattern->_histogram_max_field_greenness->getDouble());
  _histogram_black_whiteness.set(_robotPattern->_histogram_min_black_whiteness->getDouble(),_robotPattern->_histogram_max_black_whiteness->getDouble());


  _pattern_max_dist=_robotPattern->_pattern_max_dist->getDouble();
  _pattern_fit_params.fit_area_weight=_robotPattern->_pattern_fitness_weight_area->getDouble();
  _pattern_fit_params.fit_cen_dist_weight=_robotPattern->_pattern_fitness_weight_center_distance->getDouble();
  _pattern_fit_params.fit_next_dist_weight=_robotPattern->_pattern_fitness_weight_next_distance->getDouble();
  _pattern_fit_params.fit_next_dist_weight=_robotPattern->_pattern_fitness_weight_next_angle_distance->getDouble();
  _pattern_fit_params.fit_max_error=_robotPattern->_pattern_fitness_max_error->getDouble();
  _pattern_fit_params.fit_variance=sq(_robotPattern->_pattern_fitness_stddev->getDouble());
  _pattern_fit_params.fit_uniform=_robotPattern->_pattern_fitness_uniform->getDouble();

  // 读取历史缓冲帧数配置
  robot_history_.frame_count = _robotPattern->_history_buffer_frames->getInt();

  //load team image:


  if (_load_markers_from_image_file == true && _marker_image_file.length() > 0) {
    rgbImage rgbi;
    if (rgbi.load(_marker_image_file)) {
      //create a YUV lut that's based on color-labels not on custom data:
      YUVLUT minilut(4,4,4,"");
      minilut.copyChannels(*_lut3d);
      //compute a full LUT mapping based on NN-distance to color labels:
      minilut.computeLUTfromLabels();
      yuvImage yuvi;
      yuvi.allocate(rgbi.getWidth(),rgbi.getHeight());
      Images::convert(rgbi,yuvi);
      if (model.loadMultiPatternImage(yuvi,&minilut,_marker_image_rows,_marker_image_cols,_team->_robot_height->getDouble())==false) {
          fprintf(stderr,"Errors while processing team image file: '%s'.\n",_marker_image_file.c_str());
          fflush(stderr);
      }
    } else {
          fprintf(stderr,"Error loading team image file: '%s'.\n",_marker_image_file.c_str());
          fflush(stderr);
    }
    for (int i=0;i<model.getNumPatterns();i++) {
      model.getPattern(i).setEnabled(_robotPattern->_valid_patterns->isSelected(i));
    }
    model.recheckColorsUsed();
  }


}


TeamDetector::~TeamDetector()
{
  if (histogram !=0) delete histogram;
}

void TeamDetector::update(::google::protobuf::RepeatedPtrField< ::SSL_DetectionRobot >* robots, int team_color_id, int max_robots, const Image<raw8> * image, CMVision::ColorRegionList * colorlist, CMVision::RegionTree & reg_tree)
{
  color_id_team=team_color_id;
  _max_robots = max_robots;

  // 先清空结果
  robots->Clear();

    if (_unique_patterns) {
    findRobotsByModel(robots,team_color_id,image,colorlist,reg_tree); //比赛都用这个
  } else {
    findRobotsByTeamMarkerOnly(robots,team_color_id,image,colorlist); //不含方向，已经弃用，但是识别的异常稳定，不知道为什么
  }

}

void TeamDetector::updateRobotHistory(int team_color_id, const ::google::protobuf::RepeatedPtrField< ::SSL_DetectionRobot >* robots) {
    if (team_color_id == color_id_blue) {
        // Blue team
        robot_history_.blue_history.push_back(std::vector<SSL_DetectionRobot>());
        auto& current_frame = robot_history_.blue_history.back();
        
        for (int i = 0; i < robots->size(); ++i) {
            current_frame.push_back(robots->Get(i));
        }
        
        // 保持历史帧数量不超过设定值
        while (robot_history_.blue_history.size() > (size_t)robot_history_.frame_count && robot_history_.frame_count > 0) {
            robot_history_.blue_history.erase(robot_history_.blue_history.begin());
        }
    } else if (team_color_id == color_id_yellow) {
        // Yellow team
        robot_history_.yellow_history.push_back(std::vector<SSL_DetectionRobot>());
        auto& current_frame = robot_history_.yellow_history.back();
        
        for (int i = 0; i < robots->size(); ++i) {
            current_frame.push_back(robots->Get(i));
        }
        
        // 保持历史帧数量不超过设定值
        while (robot_history_.yellow_history.size() > (size_t)robot_history_.frame_count && robot_history_.frame_count > 0) {
            robot_history_.yellow_history.erase(robot_history_.yellow_history.begin());
        }
    }
}

void TeamDetector::supplementMissingRobotsFromHistory(int team_color_id, ::google::protobuf::RepeatedPtrField< ::SSL_DetectionRobot >* robots, int max_robots) {
    std::vector<std::vector<SSL_DetectionRobot>>* history_list = nullptr;
    if (robot_history_.frame_count == 0) return;
    if (team_color_id == color_id_blue) {
      // Blue team
      if (robot_history_.blue_history.empty()) return;
      history_list = &robot_history_.blue_history;
    } else if (team_color_id == color_id_yellow) {
      // Yellow team
      if (robot_history_.yellow_history.empty()) return;
      history_list = &robot_history_.yellow_history;
    } else {
      return;  // Unknown team
    }

    if (!history_list || history_list->empty()) return;
    
    // 找出历史记录中机器人数量最多的一帧
    const std::vector<SSL_DetectionRobot>* richest_frame = nullptr;
    size_t max_robots_in_history = 0;
    
    for (const auto& frame : *history_list) {
        if (frame.size() > max_robots_in_history) {
            max_robots_in_history = frame.size();
            richest_frame = &frame;
        }
    }
    
    if (!richest_frame || richest_frame->empty()) return;
    
    // 遍历最多机器人的那一帧，查找当前缺失的机器人
    for (const auto& hist_robot : *richest_frame) {
        bool found_match = false;
        
        // 检查当前检测结果中是否已存在该机器人
        for (int i = 0; i < robots->size(); ++i) {
            const auto& curr_robot = robots->Get(i);
            
            // 通过ID匹配，如果没有ID则通过位置匹配
            if ((curr_robot.has_robot_id() && hist_robot.has_robot_id() && 
                 curr_robot.robot_id() == hist_robot.robot_id()) ||
                (!curr_robot.has_robot_id() && !hist_robot.has_robot_id() &&
                 sqrt(pow(curr_robot.x() - hist_robot.x(), 2) + pow(curr_robot.y() - hist_robot.y(), 2)) < 0.05)) { // 5cm以内认为是同一个
                found_match = true;
                break;
            }
        }
        
        // 如果在当前检测结果中没有找到匹配的机器人，则从历史记录中添加
        if (!found_match && robots->size() < max_robots) {
            auto* new_robot = robots->Add();
            *new_robot = hist_robot;
            // 降低置信度表示这是从历史记录中恢复的
            new_robot->set_confidence(0.5); 
        }
    }
}

void TeamDetector::findRobotsByTeamMarkerOnly(::google::protobuf::RepeatedPtrField< ::SSL_DetectionRobot >* robots, int team_color_id, const Image<raw8> * image, CMVision::ColorRegionList * colorlist)
{
  filter_team.init( colorlist->getRegionList(team_color_id).getInitialElement() );

  //TODO: change these to update on demand:
  //local variables
  const CMVision::Region * reg=0;
  SSL_DetectionRobot * robot=0;
  while((reg = filter_team.getNext()) != 0) {
    vector2d reg_img_center(reg->cen_x,reg->cen_y);
    vector3d reg_center3d;
    _camera_params.image2field(reg_center3d,reg_img_center,_robot_height);
    vector2d reg_center(reg_center3d.x,reg_center3d.y);

    //TODO: add confidence masking:
    //float conf = det.mask.get(reg->cen_x,reg->cen_y);
    double conf=1.0;
    if (field_filter.isInFieldOrPlayableBoundary(reg_center) &&  ((_histogram_enable==false) || checkHistogram(reg,image)==true)) {
      double area = getRegionArea(reg,_robot_height);
      double area_err = fabs(area - _center_marker_area_mean);

      conf *= GaussianVsUniform(area_err, sq(_center_marker_area_stddev), _center_marker_uniform);
      //printf("-------------\n");
      if (conf < 1e-6) conf=0.0;

      /*if(unlikely(verbose > 2)){
        printf("    area=%0.1f err=%4.1f conf=%0.6f\n",area,area_err,conf);
      }
      if(det.debug) det.color(reg,rc,conf);*/

      //allow twice as many robots for now...
      //duplicate filtering will take care of the rest below:
      robot=addRobot(robots,conf,_max_robots*2);

      if (robot!=0) {
        //setup robot:
        robot->set_x(reg_center.x);
        robot->set_y(reg_center.y);
        robot->set_pixel_x(reg->cen_x);
        robot->set_pixel_y(reg->cen_y);
        robot->set_height(_robot_height);
      }
    }
  }

  // remove duplicates ... keep the ones with higher confidence:
  int size=robots->size();
  for(int i=0; i<size; i++){
    for(int j=i+1; j<size; j++){
      if(sqdist(vector2d(robots->Get(i).x(),robots->Get(i).y()),vector2d(robots->Get(j).x(),robots->Get(j).y())) < sq(_center_marker_duplicate_distance)) {
        robots->Mutable(i)->set_confidence(0.0);
      }
    }
  }

  //remove items with 0-confidence:
  stripRobots(robots);

  //remove extra items:
  while(robots->size() > _max_robots) {
    robots->RemoveLast();
  }

}







double TeamDetector::getRegionArea(const CMVision::Region * reg, double z) const {
  // calculate area of bounding box in sq mm
  vector3d a,b;
  vector2d right(reg->x2+1,reg->y2+1);
  vector2d left(reg->x1,reg->y1);
  _camera_params.image2field(a,right,z);
  _camera_params.image2field(b,left,z);
  vector3d box = a-b;

  double box_area = fabs(box.x) * fabs(box.y);
  int box_pixels = (reg->x2+1 - reg->x1) * (reg->y2+1 - reg->y1);

  // estimate world coordinate area of region
  double area = ((double)reg->area) * box_area / ((double)box_pixels);

  return(area);
}


bool TeamDetector::checkHistogram(const CMVision::Region * reg, const Image<raw8> * image) {

  if(_histogram_pixel_scan_radius == 0) return(true);

  histogram->clear();

  int ix = (int)(reg->cen_x);
  int iy = (int)(reg->cen_y);
  int num = histogram->addBox(image,ix-_histogram_pixel_scan_radius,iy-_histogram_pixel_scan_radius,
              ix+_histogram_pixel_scan_radius,iy+_histogram_pixel_scan_radius);

  float inv_num = 1.0 / num;

  float f_markeryness =
    (float)(histogram->getChannel(color_id_pink) +
            histogram->getChannel(color_id_green) +
            histogram->getChannel(color_id_cyan)) / histogram->getChannel(color_id_team);

  float f_greenness = (float)histogram->getChannel(color_id_field_green) * inv_num;

  float f_black_white =
    (float)(histogram->getChannel(color_id_white) + histogram->getChannel(color_id_black) + histogram->getChannel(color_id_clear)) * inv_num;

  /*
  if(unlikely(verbose > 0)){
    bool mky_ok = markeryness.inside(f_markeryness);
    bool grn_ok = greenness  .inside(f_greenness  );
    bool  bw_ok = black_white.inside(f_black_white);

    printf("  hist (%5.1f,%5.1f) ",reg->cen_x,reg->cen_y);

    if(verbose == 1){
      printf("mky=%0.3f grn=%0.3f baw=%0.3f [%d%d%d]\n",
             f_markeryness,f_greenness,f_black_white,
             mky_ok,grn_ok,bw_ok);
    }else{
      printf("rad=%d num=%d\n",pixel_radius,num);

      if(verbose > 2){
        printf("    cnt:  ");
        for(int i=0; i<NumColors; i++) printf(" %4d",hist[i]);
        printf("\n");

        printf("    frac: ");
        for(int i=0; i<NumColors; i++) printf(" %4d",hist[i]);
        printf("\n");
      }

      printf("    %d mky = %0.3f = %d+%d+%d / %d\n",
             mky_ok,f_markeryness,
             hist[ColorPink],hist[ColorBrightGreen],hist[ColorCyan],
             hist[team_color]);
      printf("    %d grn = %0.3f = %d / %d\n",
             grn_ok,f_greenness,
             hist[ColorFieldGreen],num);
      printf("    %d b&w = %0.3f %d/%d * %d/%d\n",
             bw_ok,f_black_white,
             hist[ColorBackground],num,
             hist[ColorWhite     ],num);
    }
  }*/

  return(
  _histogram_markeryness.inside(f_markeryness) &&
   _histogram_field_greenness.inside(f_greenness) &&
  _histogram_black_whiteness.inside(f_black_white));
}


SSL_DetectionRobot * TeamDetector::addRobot(::google::protobuf::RepeatedPtrField< ::SSL_DetectionRobot >* robots, double conf, int max_robots) {

  int size=robots->size();
  SSL_DetectionRobot * result_robot = 0;
  for (int i=0;i<size;i++) {
    if (robots->Get(i).confidence() < conf) {
      //allocate new robot at end of array
      //and shift everything down by 1...making room for newly inserted item.
      if (size < max_robots) {
        //we can expand the array by 1.
        robots->Add();
        size++;
      }
      for (int j=size-1; j>i; j--) {
        (*(robots->Mutable(j)))=robots->Get(j-1);
      }
      result_robot = robots->Mutable(i);
      result_robot->Clear();
      result_robot->set_confidence(conf);

      return result_robot;
    }
  }
  if (size < max_robots) {
    result_robot = robots->Add();
    result_robot->set_confidence(conf);
  }
  return result_robot;
}


void TeamDetector::stripRobots(::google::protobuf::RepeatedPtrField< ::SSL_DetectionRobot >* robots) {
  int size=robots->size();

  int tgt=0;
  for (int src=0;src<size;src++) {
    if (robots->Get(src).confidence() != 0.0) {
      if (tgt!=src) {
        //copy operation is needed:
        (*(robots->Mutable(tgt)))=robots->Get(src);
      }
      tgt++;
    }
  }
  for (int i=tgt;i<size;i++) {
    robots->RemoveLast();
  }
}











void TeamDetector::findRobotsByModel(::google::protobuf::RepeatedPtrField< ::SSL_DetectionRobot >* robots, int team_color_id, const Image<raw8> * image, CMVision::ColorRegionList * colorlist, CMVision::RegionTree & reg_tree)
{
  // 设置最大检测数量，限制每个颜色的最大可检测标记点数
  const int MaxDetections = _other_markers_max_detections;
  // 定义中心标记点对象，用于表示机器人的中心
  Marker cen; // center marker
  // 创建一个标记点数组，用于存储其他非中心标记点
  Marker *markers = new Marker[MaxDetections];
  // 设置查询距离，定义在区域树中查找相邻区域的最大距离
  const float marker_max_query_dist = _other_markers_max_query_distance;
  // 设置标记点间最大距离，超过此距离的标记点不会被考虑为同一机器人
  const float marker_max_dist = _pattern_max_dist;

  // 初始化过滤器，获取指定颜色的所有连通区域
  filter_team.init( colorlist->getRegionList(team_color_id).getInitialElement());
  // 定义当前处理的区域指针
  const CMVision::Region * reg=0;
  // 定义要添加的机器人对象指针
  SSL_DetectionRobot * robot=0;

  // 定义模式匹配的结果对象
  MultiPatternModel::PatternDetectionResult res;
  int num_center_markers = 0;
  // 遍历当前团队颜色的所有连通区域 filter_team就是按照队伍来的已经第一步处理之后的联通区域!
  while((reg = filter_team.getNext()) != 0) {
    // 将当前区域的图像坐标转换为中心点坐标
    vector2d reg_img_center(reg->cen_x,reg->cen_y);
    // 将图像坐标转换为场地坐标系中的3D坐标
    vector3d reg_center3d;
    _camera_params.image2field(reg_center3d,reg_img_center,_robot_height);
    // 提取场地坐标系中的2D坐标(x,y)
    vector2d reg_center(reg_center3d.x,reg_center3d.y);


    // 检查该区域是否在场地范围内（不是边界外区域）
    if (field_filter.isInFieldOrPlayableBoundary(reg_center)) {
      // 将当前区域设置为中心标记点
      ++num_center_markers;
      cen.set(reg,reg_center3d,getRegionArea(reg,_robot_height));
      // 初始化相邻标记点计数器
      int num_markers = 0;

      // 在区域树中开始查询邻近区域，以当前区域为中心，最大查询距离为marker_max_query_dist
      reg_tree.startQuery(*reg,marker_max_query_dist);
      double sd=0.0;
      CMVision::Region *mreg;
      // 遍历所有在查询距离内的相邻区域
      while((mreg=reg_tree.getNextNearest(sd))!=0 && num_markers<MaxDetections) {
        // 检查该区域是否符合"其他标记点"的过滤条件，且属于模型使用的颜色
        if(filter_others.check(*mreg) && model.usesColor(mreg->color)) {
          // 获取标记点的图像坐标
          vector2d marker_img_center(mreg->cen_x,mreg->cen_y);
          // 转换为场地坐标系中的3D坐标
          vector3d marker_center3d;
          _camera_params.image2field(marker_center3d,marker_img_center,_robot_height);
          // 创建标记点对象并初始化
          Marker &m = markers[num_markers];

          m.set(mreg,marker_center3d,getRegionArea(mreg,_robot_height));
          // 计算标记点相对于中心的距离和角度
          vector2f ofs = m.loc - cen.loc;
          m.dist = ofs.length();
          m.angle = ofs.angle();

          // 检查标记点是否在允许的最大距离内
          if(m.dist>0.0 && m.dist<marker_max_dist){
            // 如果是，则将此标记点加入数组
            num_markers++;
          }
        }
      }
      // 结束区域树查询
      reg_tree.endQuery();

      // 如果找到至少2个标记点（构成机器人必须有至少2个标记点）
      if(num_markers >= 2){
        // 根据角度对所有标记点进行排序，便于后续模式匹配
        CMPattern::PatternProcessing::sortMarkersByAngle(markers,num_markers);
        // 计算相邻标记点之间的距离和角度差
        for(int i=0; i<num_markers; i++){
          int j = (i + 1) % num_markers;  // 环形索引，最后一个元素的下一个为第一个
          markers[i].next_dist = dist(markers[i].loc,markers[j].loc);  // 计算与下一个标记点的距离
          markers[i].next_angle_dist = angle_pos(angle_diff(markers[i].angle,markers[j].angle)); // 计算与下一个标记点的角度差
        }

        // 使用多模式模型进行模式匹配，尝试识别机器人
        // 这里会执行以下步骤：
        // 1. 遍历所有可能的偏移量（处理旋转不变性）
        // 2. 为每个偏移量计算图案编码（颜色序列）
        // 3. 与预定义的机器人图案模板进行匹配
        // 4. 计算拟合误差，找到最佳匹配
        // 5. 如果匹配成功，计算机器人的位置、方向和ID
        if (model.findPattern(res,markers,num_markers,_pattern_fit_params,_camera_params))//已经debug条件断点判断这个地方几乎不会false，也就是都会检测到 
        {
          // 如果成功匹配到模式，创建一个新的机器人检测结果
          robot=addRobot(robots,res.conf,_max_robots*2);
          if (robot!=0) {
            // 设置机器人的位置信息
            robot->set_x(cen.loc.x);
            robot->set_y(cen.loc.y);
            // 如果有角度信息，设置机器人的朝向
            if (_have_angle) robot->set_orientation(res.angle);
            // 设置机器人的ID（根据模式匹配结果）
            robot->set_robot_id(res.id);
            // 设置像素坐标
            robot->set_pixel_x(reg->cen_x);
            robot->set_pixel_y(reg->cen_y);
            // 设置机器人高度
            robot->set_height(cen.height);
          }
        }
      }
    }
  }

  // 移除置信度为0的机器人检测结果
  stripRobots(robots);

  // 如果检测到的机器人数量超过最大限制，移除多余的项
  while(robots->size() > _max_robots) {
    robots->RemoveLast();
  }
  
  // 更新历史记录
  updateRobotHistory(team_color_id, robots);
  
  // 从历史记录中补全缺失的机器人
  supplementMissingRobotsFromHistory(team_color_id, robots, _max_robots);
  
  // 再次移除超出最大数量的机器人（由于补全可能超过限制）
  while(robots->size() > _max_robots) {
    robots->RemoveLast();
  }
  // std::cout<<robots->size()<<" robots detected"<<std::endl; //debug msg
  // std::cout<<"num_center_markers: "<<num_center_markers<<std::endl;//debug msg 
  // 释放之前分配的标记点数组内存
  delete[] markers;
}



};















