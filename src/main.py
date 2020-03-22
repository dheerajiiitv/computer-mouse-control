from input_feeder import InputFeeder
import face_detection 
import facial_landmark_detection 
import head_pose_estimation 
import gaze_estimation
from mouse_controller import MouseController
from argparse import ArgumentParser
import cv2
import time
import threading
def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--fd_model", required=True, type=str, help="Location of the face detection model")
    parser.add_argument("-hp", "--hp_model", required=True, type=str, help="Path to head pose estimation model")
    parser.add_argument("-ge", "--ge_model", required=True, type=str, help="Path to gaze estimation model")
    parser.add_argument("-fc", "--fc_model", required=True, type=str, help="Path to facial landmark detection model")
    parser.add_argument("-i", "--input", required=False, default=None, type=str, help="Input file")
    parser.add_argument("-p", "--precision", required=False, default='medium', type=str, help="Precision of mouse controller")
    parser.add_argument("-s", "--speed", required=False, default='medium', type=str, help="Speed of mouse controller")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str, default=None, help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will loofrom argparse import ArgumentParserk for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def main():
	
		# Grab command line args
		args = build_argparser().parse_args()
		loading_st_time = time.time()
		face_detection_model = face_detection.Model_X(model_name=args.fd_model, extensions=args.cpu_extension)
		face_detection_model.load_model()
		facial_landmark_detection_model = facial_landmark_detection.Model_X(model_name=args.fc_model, extensions=args.cpu_extension)
		facial_landmark_detection_model.load_model()
		head_pose_estimation_model = head_pose_estimation.Model_X(model_name=args.hp_model, extensions=args.cpu_extension)
		head_pose_estimation_model.load_model()
		gaze_estimation_model = gaze_estimation.Model_X(model_name=args.ge_model, extensions=args.cpu_extension)
		gaze_estimation_model.load_model()

		print("Total model loading time", (time.time() - loading_st_time))

		mouse_control  = MouseController(precision=args.precision, speed=args.speed)
		if args.input:
			print("In here")
			feed=InputFeeder(input_type='video', input_file=args.input)
		else:
			feed=InputFeeder(input_type='cam')
		feed.load_data()
		for batch in feed.next_batch():
			face, detected_qwe = face_detection_model.predict(batch)
			if face is not None:
				cv2.imshow('frame', detected_qwe)
				cv2.waitKey(5000)
				st_time = time.time()
				head_pose_estimation_angle = head_pose_estimation_model.predict(face)
				left_eye,right_eye,_ = facial_landmark_detection_model.predict(face)
				gaze_estimation_value = gaze_estimation_model.predict(left_eye, right_eye, head_pose_estimation_angle)
				mouse_control.move(gaze_estimation_value[0], gaze_estimation_value[1])
				print("Inference time, ", time.time() - st_time)
			else:
				print("No face detected in this batch")
		feed.close()




 	




if __name__ == '__main__':
	main()
