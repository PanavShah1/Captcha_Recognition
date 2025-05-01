const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const resultBox = document.getElementById("resultBox");
const pasteArea = document.getElementById("pasteArea");

let currentFile = null;

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (file) {
        currentFile = file;
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
        submitImage();
    }
});

pasteArea.addEventListener("paste", (e) => {
    const items = e.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf("image") !== -1) {
            const file = items[i].getAsFile();
            currentFile = file;
            preview.src = URL.createObjectURL(file);
            preview.style.display = "block";
            submitImage(); 
            break;
        }
    }
});

async function submitImage() {
    if (!currentFile) {
        alert("Please upload or paste an image first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", currentFile);

    try {
        const response = await fetch("http://localhost:8080/predict", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        resultBox.innerHTML = "Prediction<br><br>" + data['clean_prediction'] + "<br><br>" + data['prediction'] ;
    } catch (err) {
        resultBox.textContent = "Error occurred while decoding.";
        console.error(err);
    }
}
