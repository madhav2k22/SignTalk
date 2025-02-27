const navHomeBtn = document.querySelector("#nav-home");
const navTranslteBtn = document.querySelector("#nav-translate");
const navSettingsBtn = document.querySelector("#nav-settings");


const translateCard = document.querySelector("#translate-card");
const learnCard = document.querySelector("#learn-card");
const contributeCard = document.querySelector("#contribute-card");
const contactUsCard = document.querySelector("#contact-us-card");

//card nav
translateCard.addEventListener("click", function() {
    window.location = '/translate';
})
learnCard.addEventListener("click", function() {
    window.location = 'learn.html';
})


contactUsCard.addEventListener("click", function() {
    window.location = 'contact.html';
})
contributeCard.addEventListener("click", function() {
    window.location = 'https://github.com/madhav2k22/SignTalk.git';
})

//bottom nav
navTranslteBtn.addEventListener("click", function() {
    window.location = '/translate'
})
navSettingsBtn.addEventListener("click", function() {
    navHomeBtn.classList.remove("active-nav-btn");
    navSettingsBtn.classList.add("active-nav-btn")
    window.location = '/settings'
})