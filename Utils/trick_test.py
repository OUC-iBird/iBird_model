import PIL.Image as Image
from torchvision import transforms as transforms

im = Image.open('../maque.jpg')
new_im = transforms.ColorJitter(brightness=1)(im)
new_im = transforms.ColorJitter(contrast=1)(im)
new_im = transforms.ColorJitter(saturation=0.5)(im)
# new_im = transforms.ColorJitter(hue=0.5)(im)
new_im.save('test.jpg')