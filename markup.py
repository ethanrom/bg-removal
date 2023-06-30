def real_estate_app():
    return """
    <h3 style='text-align: center;'>Introduction</h3>
            
    <p>This app allows you to upload an image and remove its background using a custom trained model.</p>

    <h4>Information:</h4>
    <ul>
        <li><b>Very low requirements:</b> This demo is running on a free single-core CPU. The app will work on any hardware. It may take a few minutes for the first run, but afterwards, it will perform faster.</li>
        <li><b>Backend:</b> Python backend with an optimized ONNX model. No third party APIs or paid services used. Everything used is opensource</li>
        <li><b>Frontend:</b> The demo's frontend is written in Python Streamlit. Unfortunately, I cannot write a JavaScript frontend as I am a backend engineer. However, I can refer you to a colleague if you prefer a better-looking UI.</li>
        <li><b>Perspective correction:</b> Perspective correction is done via OpenCV.</li>
    </ul>
    </div>          
    """

def real_estate_app_hf():
    return """
    <div style='text-align: left;'>
    <h3 style='text-align: center;'>About this background removal demo</h3>
    <p>In this demo, the two requested features have been split into two separate tabs for better individual evaluation. They can be combined into one in the final submission.</p>
    <br>
    <h4>How to use:</h4>
    <ul>
    <li><b>Remove Background:</b> Use this tab to remove the background of the images. Right-click and save the image once done.</li>
    <li><b>Correct Perspective:</b> Use this tab to check the perspective correction option. Please make sure to use an image with the background already removed.</li>
    </ul>
    <br>
    <p><b>Update 1</b>
    <p>I understand that while the app performs well in some cases, there are instances where it doesn't work correctly. Achieving perfect background removal in all cases is indeed a challenging task, which is why paid API services exist for this purpose. 
    Training a more accurate model would require a substantial number of images, incur higher costs, and take a longer time. Additionally, using such a model without a GPU would not be feasible.</p>
    <p>To address the limitations of the current version, I have implemented RGB sliders and curves in the latest update. These adjustments can be used to manually fine-tune the results. 
    The objective is to find settings that create a clear distinction between the foreground and background by manipulating the colors using the sliders and curves.</p>
    <p>I will also explore the possibility of incorporating better preprocessing algorithms when I have some free time. However, please note that the primary constraint lies in the model itself, and significant improvements might not be achievable through algorithmic changes alone.</p>
    </div>
    """

def sliders_intro():
    return """
    <p>Newly added sliders which will appear after an image is uploaded serve as interactive tools to adjust various aspects of the uploaded image. Each slider corresponds to a specific color channel (red, green, blue) or a curve adjustment. 
    By using these sliders, users can fine-tune the color levels and apply curve modifications to achieve the desired visual effect.</p>
    <p>For the RGB Adjustments section, users can use the Red, Green, and Blue sliders to set the minimum and maximum values for each color channel. 
    By adjusting these values, users can enhance or reduce the intensity of each color channel individually, allowing for precise color adjustments.</p>
    <p>In the Curves Adjustment section, users can utilize the Red Curve, Green Curve, and Blue Curve sliders to control the brightness of the respective color channels. 
    By moving these sliders, users can create custom curves, influencing the overall tone and contrast of the image.</p>
    <p>The Masking section offers the Threshold slider, which determines the cutoff point for the transparency mask. 
    Users can adjust the threshold value to define the boundary between foreground and background elements. This feature enables users to isolate objects by selectively applying transparency to specific areas of the image.</p>
    """

def perspective_intro():
    return """
    there are two different perspective correction methods you can chose from, the difference is how they determine the transformation matrix used for warping the image.

    In the Four-Point Perspective Correction, the method uses a four-point perspective transform. It first detects lines in the image using Hough Lines, and then calculates the endpoints of these lines. 
    If enough endpoints are found (at least 4), a convex hull is created based on these endpoints. From the convex hull, a four-sided polygon is obtained, representing the region of interest. 
    The width and height of this polygon are determined, and a destination set of points is defined to which the polygon will be mapped. 
    Finally, a perspective transformation matrix is computed, and the image is warped accordingly.

    In the Convex Hull Homography Perspective Correction, a similar process is followed, but instead of using a four-point perspective transform, it uses a homography transform. 
    After obtaining the endpoints, a convex hull is created, and a four-sided polygon is extracted from it. The width and height of this polygon are calculated, and a destination set of points is defined. 
    But Homography function is used to compute the homography matrix. 
    This matrix defines the transformation between the source polygon and the destination polygon, and the image is warped using the warpPerspective function.
    This method is more accurate and more robust at the cost of being more resource-intensive

    The parameters threshold value, min line length, and max line gap in both methods control the detection of lines in the image. 
    These parameters affect the number and quality of lines detected, which in turn can impact the accuracy of the perspective correction. 
    Adjusting these values allows fine-tuning the perspective correction process based on the specific characteristics of the input image. 
    However, it is important to note that changing these values requires some experimentation to achieve the desired results, and it is recommended to find the optimal values through trial and error.
    """
