from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.generics import ListAPIView,ListCreateAPIView
# Create your views here.
@csrf_exempt 
def test(request):

    from skimage.io import imread
    from skimage.color import rgb2hsv, rgb2lab
    import numpy as np
    from scipy.signal import find_peaks
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


    def changeRes(img, pixel_height):
        res_divider = round(np.shape(img)[0]/pixel_height)
        img = img[::res_divider, ::res_divider, :]
        return img

    def splitImage(img):
        img_lab = rgb2lab(img/255)
        v_light = np.average(img_lab[:, :, 0], axis = 0)
        h_light = np.average(img_lab[:, :, 0], axis = 1)
        bound_inc = 1
        peak1_sat_bound = -100
        target_num_peaks1 = 1

        while True:
            peaks1 = np.asarray(find_peaks(-v_light, height = [peak1_sat_bound, 0], threshold = None, distance = .045*np.size(v_light))[0])
            num_peaks1 = np.size(peaks1) 
            if num_peaks1 == target_num_peaks1:
                break
            else:
                peak1_sat_bound += bound_inc
        peak2_sat_bound = -100
        target_num_peaks2 = 1 
        
        while True:
            peaks2 = np.asarray(find_peaks(-h_light, height = [peak2_sat_bound, 0], threshold = None, distance = .045*np.size(h_light))[0])
            num_peaks2 = np.size(peaks2) 
            if num_peaks2 == target_num_peaks2:
                break
            else:
                peak2_sat_bound += bound_inc
        strip = img[:, 0:peaks1[0], :]

        chart = img[peaks2[0]:np.shape(img)[0], (peaks1[0] + 1):np.shape(img)[1], :]

        # print("strip: ", strip,chart)

        return strip, chart
        
    def cropTop(img, pct):
        img = img[round(pct*np.shape(img)[0]):np.shape(img)[0], :, :]
        return img
    
    def findStripColors(img, target_num_peaks, crop_pct):
        img = cropTop(img, crop_pct)
        saturation = rgb2hsv(img)[:, :, 1]
        saturation = np.average(saturation, axis = 1)
        bound_inc = .001
        peak_sat_bound = 0
        while True:
            peaks = np.asarray(find_peaks(saturation, height = [peak_sat_bound, 1], threshold = None, distance = .045*np.size(saturation))[0])
            num_peaks = np.size(peaks)
            if num_peaks == target_num_peaks:
                break
            else:
                peak_sat_bound += bound_inc
        img_avg_peaks = img[peaks, int(np.shape(img)[1]/2), :]
        img_avg_peaks = np.reshape(img_avg_peaks, (10, 1, 3))
        peak_colors = img_avg_peaks.astype('uint8')
        return peak_colors

    def cropBottom(img, pct):
        img = img[0:round((1 - pct)*np.shape(img)[0]), :, :]
        return img

    def findChartColors(img, num_peaks_vert, num_peaks_horz, pct_top, pct_bottom):
        img = cropTop(img, pct_top)
        img = cropBottom(img, pct_bottom)
        chart_hsv = rgb2hsv(img/255)
        v_chart_sat = np.average(chart_hsv[:, :, 1], axis = 1)
        h_chart_sat = np.average(chart_hsv[:, :, 1], axis = 0)
        bound_inc = .001
        peak3_sat_bound = 0
        target_num_peaks3 = num_peaks_vert
        while True:
            peaks3 = np.asarray(find_peaks(v_chart_sat, height = [peak3_sat_bound, 1], threshold = None, distance = .045*np.size(v_chart_sat))[0])
            num_peaks3 = np.size(peaks3)
            if num_peaks3 == target_num_peaks3:
                break
            else:
                peak3_sat_bound += bound_inc
        peak4_sat_bound = 0
        target_num_peaks4 = num_peaks_horz
        while True:
            peaks4 = np.asarray(find_peaks(h_chart_sat, height = [peak4_sat_bound, 1], threshold = None, distance = .06*np.size(h_chart_sat))[0])
            num_peaks4 = np.size(peaks4)
            if num_peaks4 == target_num_peaks4:
                break
            else:
                peak4_sat_bound += bound_inc
        analyte_chart = np.zeros((target_num_peaks3, target_num_peaks4, 3))
        for i in range(target_num_peaks3):
            for j in range(target_num_peaks4):
                analyte_chart[i, j, :] = img[peaks3[i], peaks4[j], :]
        analyte_chart = analyte_chart.astype('uint8')
        return analyte_chart

    if request.method == 'POST':
        image = request.FILES['image']
        print("Image: ",image)

        img = imread(image)
        # img = imread('with_flash.jpeg')
        img = changeRes(img, 256)
        fig, ax = plt.subplots()
        ax.imshow(img) 

        # 2. Split image into strip and chart components 

        strip, chart = splitImage(img) 
        fig3, ax3 = plt.subplots(1, 2)
        ax3[0].imshow(strip)
        ax3[1].imshow(chart)

        # 3. Separate out test strip colors

        strip_colors = findStripColors(strip, 10, 0) 
        fig5, ax5 = plt.subplots(1, 2)
        ax5[0].imshow(strip_colors) 

        # 4. Separate out calibration chart colors

        chart_colors = findChartColors(chart, 10, 7, 0.01, 0.06)
        ax5[1].imshow(chart_colors)
        # plt.show()
        response = HttpResponse(content_type='image/png')
        plt.savefig(response)
        return response
        # return HttpResponse("Working")

    return HttpResponse("Testing")



class ImamgeApi(ListAPIView):
    @csrf_exempt 
    def post(self,request):
        

        from skimage.io import imread
        from skimage.color import rgb2hsv, rgb2lab
        import numpy as np
        from scipy.signal import find_peaks
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt


        def changeRes(img, pixel_height):
            res_divider = round(np.shape(img)[0]/pixel_height)
            img = img[::res_divider, ::res_divider, :]
            return img

        def splitImage(img):
            img_lab = rgb2lab(img/255)
            v_light = np.average(img_lab[:, :, 0], axis = 0)
            h_light = np.average(img_lab[:, :, 0], axis = 1)
            bound_inc = 1
            peak1_sat_bound = -100
            target_num_peaks1 = 1

            while True:
                peaks1 = np.asarray(find_peaks(-v_light, height = [peak1_sat_bound, 0], threshold = None, distance = .045*np.size(v_light))[0])
                num_peaks1 = np.size(peaks1) 
                if num_peaks1 == target_num_peaks1:
                    break
                else:
                    peak1_sat_bound += bound_inc
            peak2_sat_bound = -100
            target_num_peaks2 = 1 
            
            while True:
                peaks2 = np.asarray(find_peaks(-h_light, height = [peak2_sat_bound, 0], threshold = None, distance = .045*np.size(h_light))[0])
                num_peaks2 = np.size(peaks2) 
                if num_peaks2 == target_num_peaks2:
                    break
                else:
                    peak2_sat_bound += bound_inc
            strip = img[:, 0:peaks1[0], :]

            chart = img[peaks2[0]:np.shape(img)[0], (peaks1[0] + 1):np.shape(img)[1], :]

            # print("strip: ", strip,chart)

            return strip, chart
            
        def cropTop(img, pct):
            img = img[round(pct*np.shape(img)[0]):np.shape(img)[0], :, :]
            return img
        
        def findStripColors(img, target_num_peaks, crop_pct):
            img = cropTop(img, crop_pct)
            saturation = rgb2hsv(img)[:, :, 1]
            saturation = np.average(saturation, axis = 1)
            bound_inc = .001
            peak_sat_bound = 0
            while True:
                peaks = np.asarray(find_peaks(saturation, height = [peak_sat_bound, 1], threshold = None, distance = .045*np.size(saturation))[0])
                num_peaks = np.size(peaks)
                if num_peaks == target_num_peaks:
                    break
                else:
                    peak_sat_bound += bound_inc
            img_avg_peaks = img[peaks, int(np.shape(img)[1]/2), :]
            img_avg_peaks = np.reshape(img_avg_peaks, (10, 1, 3))
            peak_colors = img_avg_peaks.astype('uint8')
            return peak_colors

        def cropBottom(img, pct):
            img = img[0:round((1 - pct)*np.shape(img)[0]), :, :]
            return img

        def findChartColors(img, num_peaks_vert, num_peaks_horz, pct_top, pct_bottom):
            img = cropTop(img, pct_top)
            img = cropBottom(img, pct_bottom)
            chart_hsv = rgb2hsv(img/255)
            v_chart_sat = np.average(chart_hsv[:, :, 1], axis = 1)
            h_chart_sat = np.average(chart_hsv[:, :, 1], axis = 0)
            bound_inc = .001
            peak3_sat_bound = 0
            target_num_peaks3 = num_peaks_vert
            while True:
                peaks3 = np.asarray(find_peaks(v_chart_sat, height = [peak3_sat_bound, 1], threshold = None, distance = .045*np.size(v_chart_sat))[0])
                num_peaks3 = np.size(peaks3)
                if num_peaks3 == target_num_peaks3:
                    break
                else:
                    peak3_sat_bound += bound_inc
            peak4_sat_bound = 0
            target_num_peaks4 = num_peaks_horz
            while True:
                peaks4 = np.asarray(find_peaks(h_chart_sat, height = [peak4_sat_bound, 1], threshold = None, distance = .06*np.size(h_chart_sat))[0])
                num_peaks4 = np.size(peaks4)
                if num_peaks4 == target_num_peaks4:
                    break
                else:
                    peak4_sat_bound += bound_inc
            analyte_chart = np.zeros((target_num_peaks3, target_num_peaks4, 3))
            for i in range(target_num_peaks3):
                for j in range(target_num_peaks4):
                    analyte_chart[i, j, :] = img[peaks3[i], peaks4[j], :]
            analyte_chart = analyte_chart.astype('uint8')
            return analyte_chart

        
        image = request.FILES['image']
        print("Image: ",image)

        img = imread(image)
        # img = imread('with_flash.jpeg')
        img = changeRes(img, 256)
        fig, ax = plt.subplots()
        ax.imshow(img) 

            # 2. Split image into strip and chart components 

        strip, chart = splitImage(img) 
        fig3, ax3 = plt.subplots(1, 2)
        ax3[0].imshow(strip)
        ax3[1].imshow(chart)

            # 3. Separate out test strip colors

        strip_colors = findStripColors(strip, 10, 0) 
        fig5, ax5 = plt.subplots(1, 2)
        ax5[0].imshow(strip_colors) 

            # 4. Separate out calibration chart colors

        chart_colors = findChartColors(chart, 10, 7, 0.01, 0.06)
        ax5[1].imshow(chart_colors)
            # plt.show()
        response = HttpResponse(content_type='image/png')
        plt.savefig(response)
        return response

        