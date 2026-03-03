 #pragma once
 #include <vector>
 
 namespace Yolo {
 
 struct Candidate {
   int x1;
   int y1;
   int x2;
   int y2;
   float conf;
   int class_id;
 };
 
 struct CandidateSet {
  std::vector<Candidate> robots;
  std::vector<Candidate> robots_blue;
  std::vector<Candidate> robots_yellow;
   std::vector<Candidate> balls;
 };
 
 }
