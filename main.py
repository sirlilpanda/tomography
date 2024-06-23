from PIL import Image 
import matplotlib.pyplot as plt 
import numpy as np
from pprint import pprint
import scipy

def norm(data):
    data_diff = (float(max(data))-float(min(data)))
    return np.array([2*(float(x))/(data_diff) for x in data])

def get_roations(image, roations) -> list[np.array]:
    out = []
    # yes i can mirror around 180 degrees however this is easier
    for angle in np.linspace(0, 360, roations, endpoint=False):       
        out.append(np.array(image.rotate(angle, fillcolor=(0,0,0)), dtype=np.float32))
    return out

def make_sinogram(rotated_images : list[np.array]):
    out = []
    for image in rotated_images:
        temp = np.sum(image, axis=0, dtype=np.float32)[:, 1]
        out.append(temp)
    return np.array(out)

def filter_back_projection(mags, cutoff=0): 
    filter_size = mags[0].size
    print(filter_size) 
    #cutoff here as if an image is large enough this is important to 
    #remove high freq rippling 
    if not cutoff: cutoff = filter_size
    scaling = 1
    filter_coeffs = np.zeros(cutoff//2)
    #should be 0 and 2pi but i found this tends to work better
    w = np.linspace(-np.pi, np.pi, filter_size)
    for i in range(0, cutoff//2):
        filter_coeffs[i] = w[i]*scaling
    filter_coeffs = np.concatenate((filter_coeffs[::-1],filter_coeffs[0:]))
    remaining = (filter_size-filter_coeffs.size)//2
    filter_coeffs = -np.concatenate((np.full(remaining, w[0]), filter_coeffs, np.full(remaining, w[0])))
    # all code above creates the ramp filter

    plt.figure()
    plt.plot(filter_coeffs)

    out = []
    for i in mags:
        ft = np.fft.fft(i)
        out.append(np.fft.ifft(ft*filter_coeffs).real)
        
    plt.figure()
    plt.title("ramp filter applied to the first slice")
    plt.plot(out[0], label="filtered")
    plt.plot(mags[0], label="pre filtered")
    plt.xlabel("sample (unitless)")
    plt.xlabel("amplitude (unitless)")
    plt.grid(True)
    plt.legend()
    #this might be a negitive as my filter is (-pi, pi) instead of 0, 2pi
    #but my code doesnt like 0, 2pi for the filter
    return np.array(out)



def rebuild_image(mags):
                        #yes i know this isnt a good way to do it numpy is hard
    arrays = np.array([np.zeros(mags[0].size) for x in range(0, mags[0].size)])
    angles = np.linspace(0, 360, len(mags))

    for i, mag in enumerate(reversed(mags)):
        image = np.stack([mag for x in range(0, mag.size)])
        img = scipy.ndimage.rotate(image, angles[i], reshape=False)
        print(i)
        arrays+=np.array(img)

    plt.figure()
    ax = plt.gca()    
    plt.title("reconstructed")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(arrays, interpolation="none", cmap="gist_gray" )
    

def main():
    Original_Image = Image.open("SheppLogan_Phantom.png") 
    angles = 1024
    # Original_Image = Image.open("basic.png") 
    temp = make_sinogram(get_roations(Original_Image, angles))
    print(temp[0])
    plt.figure()
    plt.title("sinogram")
    plt.xlabel("intensity (unitless)")
    plt.ylabel("sampled angles from 0, 360 (degrees)")
    plt.imshow(temp, interpolation="none", cmap="gist_gray" )
    rebuild_image(temp)
    filtered = filter_back_projection(temp)
    rebuild_image(filtered)
    plt.figure()
    plt.title("backfiltered")
    plt.xlabel("intensity (unitless)")
    plt.ylabel("sampled angles from 0, 360 (degrees)")
    plt.imshow(filtered, interpolation="none", cmap="gray" )
    plt.show()
    



if __name__ == "__main__":
    main()
