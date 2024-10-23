class AuthService {
    constructor() {
        this.auth = firebase.auth();
        this.setupAuthStateListener();
    }

    // Listen for auth state changes
    setupAuthStateListener() {
        this.auth.onAuthStateChanged(user => {
            if (user) {
                this.showUserDashboard(user);
            } else {
                this.showLoginForm();
            }
        });
    }

    // Sign up with email and password
    async signUp(email, password) {
        try {
            await this.auth.createUserWithEmailAndPassword(email, password);
        } catch (error) {
            alert(error.message);
        }
    }

    // Sign in with email and password
    async signIn(email, password) {
        try {
            await this.auth.signInWithEmailAndPassword(email, password);
        } catch (error) {
            alert(error.message);
        }
    }

    // Sign in with Google
    async signInWithGoogle() {
        try {
            const provider = new firebase.auth.GoogleAuthProvider();
            await this.auth.signInWithPopup(provider);
        } catch (error) {
            alert(error.message);
        }
    }

    // Sign out
    async signOut() {
        try {
            await this.auth.signOut();
        } catch (error) {
            alert(error.message);
        }
    }

    // UI Helper methods
    showUserDashboard(user) {
        document.getElementById('loginSection').classList.add('hidden');
        document.getElementById('signupSection').classList.add('hidden');
        document.getElementById('userDashboard').classList.remove('hidden');
        document.getElementById('userEmail').textContent = `Logged in as: ${user.email}`;
    }

    showLoginForm() {
        document.getElementById('loginSection').classList.remove('hidden');
        document.getElementById('signupSection').classList.add('hidden');
        document.getElementById('userDashboard').classList.add('hidden');
    }

    showSignupForm() {
        document.getElementById('loginSection').classList.add('hidden');
        document.getElementById('signupSection').classList.remove('hidden');
        document.getElementById('userDashboard').classList.add('hidden');
    }
}
