import numpy as np
import os
import cv2

class EducationalChallengeImageColorizationEvaluator:
  def __init__(self, ground_truth_images_dir, round=1):
    """
    `round` : Holds the round for which the evaluation is being done. 
    can be 1, 2...upto the number of rounds the challenge has.
    Different rounds will mostly have different ground truth files.
    """
    self.ground_truth_images_dir = ground_truth_images_dir
    self.round = round

  def _evaluate(self, client_payload, _context={}):
    """
    `client_payload` will be a dict with (atleast) the following keys :
      - submission_dir_path : local directory where the submission files will be stored
      - aicrowd_submission_id : A unique id representing the submission
      - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
    """

    submission_dir_path = client_payload["submission_dir_path"]
    aicrowd_submission_id = client_payload["aicrowd_submission_id"]
    aicrowd_participant_uid = client_payload["aicrowd_participant_id"]

    ground_truth_images = []
    for root, dirs, files in os.walk(self.ground_truth_images_dir):
      for file in files:
        if (file.endswith(".jpg") or
            file.endswith(".jpeg")):
          ground_truth_images.append(os.path.join(root, file))
        else:
          raise Exception("Ground truth dir contains invalid file(s). Unkown file: '{}'".
              format(file))

    submission_images = []
    for root, dirs, files in os.walk(submission_dir_path):
      for file in files:
        if (file.endswith(".jpg") or
            file.endswith(".jpeg")):
          submission_images.append(os.path.join(root, file))
        else:
          raise Exception("Submission dir contains invalid file(s). Unkown file: '{}'".
              format(file))

    if len(submission_images) != len(ground_truth_images):
      raise Exception("Submissions contains {} images, while we expected {} image.".
          format(len(submission_images), len(ground_truth_images)))

    ground_truth_images.sort()
    submission_images.sort()

    scores = []
    for ground_truth_image, submission_image in zip(ground_truth_images, submission_images):
      if os.path.basename(ground_truth_image) != os.path.basename(submission_image):
        raise Exception("Invalid submission file {}. Expected {}.".
            format(os.path.basename(submission_image), os.path.basename(ground_truth_image)))
      else:
        scores.append(self.compute_score(submission_image, ground_truth_image))

    scores = np.array(scores)

    _result_object = {
        "score": scores.mean(),
        "score_secondary" : scores.mean()
    }

    return _result_object

  def compute_score(self, submission_image_path, ground_truth_image_path):
    # NOTE: opencv uses BGR encoding. Does not matter as long as it's consistent
    submission_image = cv2.imread(submission_image_path)
    if submission_image is None:
      raise Exception("Error reading submission image '{}'".
          format(os.path.basename(submission_image_path)))

    ground_truth_image = cv2.imread(ground_truth_image_path)
    if ground_truth_image is None:
      raise Exception("Error reading ground truth image '{}'".
          format(os.path.basename(ground_truth_image_path)))


    if ground_truth_image.shape != submission_image.shape:
      raise Exception("The image size does not match. Expected '{}', actual '{}'".
          format(ground_truth_image.shape, submission_image.shape))

    # Mean Squared Error (MSE)
    return ((submission_image - ground_truth_image) ** 2).mean()


if __name__ == "__main__":

    SUBMISSION_DIR = "data/output_images/"
    GROUND_TRUTH_IMAGES = "data/ground_truth_images/"

    ground_truth_images_dir = "data/ground_truth_images/"

    _client_payload = {}
    _client_payload["submission_dir_path"] = "data/output_images/"
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234

    # Instantiate a dummy context
    _context = {}
    # Instantiate an evaluator
    aicrowd_evaluator = \
        EducationalChallengeImageColorizationEvaluator(ground_truth_images_dir, round=1)
    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print(result)
