const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const recordButton = document.getElementById("record");
const resultImg = document.getElementById("result");
const saveImage = document.getElementById("save_image");
const Imagelabel = document.getElementById("image_label");
const results = document.getElementById("results")
const boundingReact = {
    W: 160,
    H: 160,
    X: (canvas.width - 160) / 2,
    Y: (canvas.height - 160) / 2,
};

function createMessage(status, message){
    const spanEle = document.createElement('span');
    console.log(status)
    spanEle.style.color = status === "error"? "red": "green"
    spanEle.innerText = message
    return spanEle
}

function appendCroppedCanvasToForm(formData) {
    const croppedCanvas = document.createElement("canvas");
    const croppedContext = croppedCanvas.getContext("2d");
    croppedCanvas.width = boundingReact.W;
    croppedCanvas.height = boundingReact.H;
    croppedContext.drawImage(
        canvas,
        boundingReact.X,
        boundingReact.Y,
        boundingReact.W,
        boundingReact.H,
        0,
        0,
        boundingReact.W,
        boundingReact.H
    );

    const label = Imagelabel.value;
    formData.append(Imagelabel.name, label);

    return new Promise((resolve) => {
        croppedCanvas.toBlob((blob) => {
            if (blob) {
                filename = `${label}_${Date.now()}.jpg`
                formData.append("image", blob, filename);
                resolve(formData); 
            } else {
                console.error("Failed to create blob");
            }
        }, "image/jpeg");
    });
}

let recording = false;

navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;

        video.addEventListener("play", () => {
            const drawRectangle = () => {
                context.drawImage(video, 0, 0, canvas.width, canvas.height);

                context.strokeStyle = "blue";
                context.lineWidth = 2;
                context.strokeRect(boundingReact.X, boundingReact.Y, boundingReact.W, boundingReact.H);

                if (!recording) requestAnimationFrame(drawRectangle);
            };

            drawRectangle();
        });
    })
    .catch((err) => {
        console.error("Không thể truy cập camera: ", err);
    });

saveImage.addEventListener("click", async () => {
    let formData = new FormData();
    formData = await appendCroppedCanvasToForm(formData);
    try {
        const response = await fetch("http://127.0.0.1:5000/save_image", {
            method: "POST",
            body: formData,
        });
        if (response.ok) {
            const data = await response.json();

            console.log(data)

            const messageElement = createMessage(data.status, data.message);
            results.appendChild(messageElement);
        } else {
            console.error("Lỗi xử lý ảnh:", await response.text());
        }
    } catch (err) {
        const messageElement = createMessage("error", "Lỗi kết nối đến server");
        results.appendChild(messageElement);
    }
});
