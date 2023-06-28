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
    </div>
    """