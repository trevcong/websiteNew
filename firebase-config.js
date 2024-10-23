// website/firebase-config.js
const firebaseConfig = {
    apiKey: "AIzaSyBOt-v2aqoop_B9iF8wkGq-T2lIPNkdk8k",
    authDomain: "websitedb-a1827.firebaseapp.com",
    databaseURL: "https://websitedb-a1827-default-rtdb.firebaseio.com",
    projectId: "websitedb-a1827",
    storageBucket: "websitedb-a1827.appspot.com",
    messagingSenderId: "976408279985",
    appId: "1:976408279985:web:d86e3bfce978fec9175b90",
    measurementId: "G-ZHBCKS0JKY"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const auth = firebase.auth();
