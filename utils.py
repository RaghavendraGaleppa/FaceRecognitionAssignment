import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

class CenterFaceAlign(object):

	def __init__(self):
		self.mtcnn = MTCNN(post_process=False)


	def __call__(self,image,to_tensor=False):
		return self.mtcnn(image).detach().numpy().squeeze()

class FaceMatch(object):

	def __init__(self, PATH):
		
		self.model = InceptionResnetV1(pretrained='vggface2',classify=True,
				num_classes=961)
		saved_checkpoint = torch.load(PATH)
		self.model.classify = False
		self.model.load_state_dict(saved_checkpoint['state_dict'])
		self.model.eval()
		# load model	

	def __call__(self, image_1, image_2):
		out = self.model(torch.stack(image_1, image_2)).detach()
		distance = (out[0] - out[1]).norm()
		if distance < 0.8:
			print(f"The faces match")
		else:
			print(f"Different Faces")

		print(f"Distance between the images: {distance}")


