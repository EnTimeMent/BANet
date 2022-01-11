# BANet

## Introduction
This is for the paper 'Learning Bodily and Temporal Attention in Protective Movement Behavior Detection', presented at ACIIW'19.

If you want to run the network on EmoPain dataset (http://www.emo-pain.ac.uk/), you can choose to operate the model on
   i) the 13 joint angles, 13 energies data, and 4 sEMG data.
   ii) the raw coordinates.

If you want to run the network on your own dataset, please make sure you understand how the BANet works, 
and the best way to do this is reading the paper: 

Wang, Chongyang, Min Peng, Temitayo A. Olugbade, Nicholas D. Lane, Amanda C. De C. Williams, and Nadia Bianchi-Berthouze. "[Learning temporal and bodily attention in protective movement behavior detection](https://ieeexplore.ieee.org/abstract/document/8925084/)", 2019 8th International Conference on Affective Computing and Intelligent Interaction Workshops and Demos (ACIIW), pp. 324-330. IEEE, 2019.

Then you can change the number of body parts/sensors to your need.

## Code Description
- BANet_angle.py is the proposed learning model with two attention mechanisms, namely bodily attention and temporal attention, which takes joint angles, energies, and sEMG data as input.

- BANet_coordinate.py s the proposed learning model with two attention mechanisms, namely bodily attention and temporal attention, which takes the 22 sets of coordinates (excluding the 4 sets of coordinates collected from both feet as the data) as input.

- BANet_body.py is the variant of BANet_angle that only has bodily attention.

- BANet_time.py is the variant of BANet_angle that only has temporal attention.

(Within each code, instructions are also provided.)

## Citation
If you find anything useful, consider citing
@inproceedings{wang2019learning,
  title={Learning temporal and bodily attention in protective movement behavior detection},
  author={Wang, Chongyang and Peng, Min and Olugbade, Temitayo A and Lane, Nicholas D and Williams, Amanda C De C and Bianchi-Berthouze, Nadia},
  booktitle={2019 8th International Conference on Affective Computing and Intelligent Interaction Workshops and Demos (ACIIW)},
  pages={324--330},
  year={2019},
  organization={IEEE}
}
