import evaluate
from PIL import Image
import json
import numpy as np
import cv2


class PosedImage:
    def __init__(self, json_line):
        img_dict = json.loads(json_line)
        self.pose = np.array(img_dict["pose"])
        self.img_name = img_dict["imgfname"]
        

    def write_bearings(self, neuralnet, bearings_file, folder_name=""):
        # Obtain neural net output
        img = Image.open(folder_name+self.img_name)
        heatmap = neuralnet.sliding_window(img)

        # Compute animal bearings here and save to self.animals.
        # Next, you can use all this information to triangulate the animals!
        
        bearings = {}
        # For example, finding the llamas:
        # if np.any(heatmap == 2.0):
        #     llama_coords = np.where(heatmap == 2.0)
        #     average_llama = np.mean(llama_coords, axis=1)
        #     bearings["llama"] = ...
        # Now you need to convert this to a horizontal bearing as an angle.
        # Use the camera matrix for this!

        # There are ways to get much better bearings.
        # Try and think of better solutions than just averaging.

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        
        # Change thresholds
        params.minThreshold = 1
        params.maxThreshold = 5
        params.thresholdStep = 1
        
        # Filter by Area.
        params.filterByArea = True

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(params)
        else : 
            detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(heatmap)

        for animal in bearings:
            bearing_dict = {"pose":self.pose.tolist(),
                            "animal":animal,
                            "bearing":bearings[animal]}
            bearing_line = json.dumps(bearing_dict)


if __name__ == "__main__":
    # Set up the network
    exp = evaluate.Evaluate()

    # Read in the images
    images_fname = "../system_output/images.txt"
    with open(images_fname, 'r') as images_file:
        posed_images = [PosedImage(line) for line in images_file]

    # Compute bearings and write to file
    bearings_fname = "../system_output/bearings.txt"
    with open(bearings_fname, 'w') as bearings_file:
        for posed_image in posed_images:
            posed_image.write_bearings(exp, bearings_file, "../system_output/")

    