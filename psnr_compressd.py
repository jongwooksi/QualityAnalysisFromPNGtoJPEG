import cv2
import os
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt

def cal_ssim(grayA, grayB):
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    return score

def valuelist(size):
    object = list()
    for i in range(size):
        object.append( list() ) 
    return object

file_style = os.listdir('/home/jwsi/compressedImage/원본파일/')
file_style.sort()

file_content = os.listdir('/home/jwsi/compressedImage/50%압축/')
file_content.sort()

file_content2 = os.listdir('/home/jwsi/compressedImage/90%압축/')
file_content2.sort()

count = 0

psnr_content = [0 for i in range(6)]
ssim_content = [0 for i in range(6)]
values = valuelist(12)


for i in range(140): 
    img1 = cv2.imread('/home/jwsi/compressedImage/원본파일/'+ file_style[i])
    img2 = cv2.imread('/home/jwsi/compressedImage/50%압축/'+file_content[i])
    img3 = cv2.imread('/home/jwsi/compressedImage/80%압축/'+file_content[i])
    img4 = cv2.imread('/home/jwsi/compressedImage/90%압축/'+file_content2[i])
    img5 = cv2.imread('/home/jwsi/compressedImage/95%압축/'+file_content[i])
    img6 = cv2.imread('/home/jwsi/compressedImage/98%압축/'+file_content[i])

    psnr_content[0] += cv2.PSNR(img1, img1)
    psnr_content[1] += cv2.PSNR(img1, img2)
    psnr_content[2] += cv2.PSNR(img1, img3)
    psnr_content[3] += cv2.PSNR(img1, img4)
    psnr_content[4] += cv2.PSNR(img1, img5)
    psnr_content[5] += cv2.PSNR(img1, img6)

    values[0].append( cv2.PSNR(img1, img1))
    values[1].append( cv2.PSNR(img1, img2))
    values[2].append( cv2.PSNR(img1, img3))
    values[3].append( cv2.PSNR(img1, img4))
    values[4].append( cv2.PSNR(img1, img5))
    values[5].append( cv2.PSNR(img1, img6))
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
    img6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)

    ssim_content[0] += cal_ssim(img1, img1)
    ssim_content[1] += cal_ssim(img1, img2)
    ssim_content[2] += cal_ssim(img1, img3)
    ssim_content[3] += cal_ssim(img1, img4)
    ssim_content[4] += cal_ssim(img1, img5)
    ssim_content[5] += cal_ssim(img1, img6)
    
    values[6].append(cal_ssim(img1, img1))
    values[7].append(cal_ssim(img1, img2))
    values[8].append(cal_ssim(img1, img3))
    values[9].append(cal_ssim(img1, img4))
    values[10].append(cal_ssim(img1, img5))
    values[11].append(cal_ssim(img1, img6))

    count += 1


contentloss = [x/count for x in psnr_content]
contentloss2 = [x/count for x in ssim_content]

for i in range(6):  
    print("{:>7.3f}".format(contentloss[i]), end= "        ")
print()

for i in range(6):  
    print("{:>7.3f}".format(contentloss2[i]), end= "        ")
print()

plt.boxplot(values[1:6])
plt.xlabel("Compress Percent")
plt.ylabel("PSNR")
plt.scatter([1,2,3,4,5], contentloss[1:], color='r')
plt.xticks([1,2,3,4,5], ['50%','80%','90%','95%','98%'])
plt.show()

plt.boxplot(values[7:])
plt.xlabel("Compress Percent")
plt.ylabel("SSIM")
plt.scatter([1,2,3,4,5], contentloss2[1:], color='r')
plt.xticks([1,2,3,4,5], ['50%','80%','90%','95%','98%'])
plt.show()