function analyse() {
    let textarea = document.getElementById("textarea");
    let review = textarea.value;
    window.open("/analyse/" + review, target="_self");
}