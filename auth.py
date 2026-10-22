
import streamlit as st
from supabase import create_client


def get_supabase_client():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "")

    if not url or not key:
        st.error(
            "Supabase settings are missing. "
            "Check Streamlit Secrets."
        )
        st.stop()

    return create_client(url, key)


def clear_saved_session():
    st.session_state.pop("sb_access_token", None)
    st.session_state.pop("sb_refresh_token", None)
    st.session_state.pop("sb_user_id", None)
    st.session_state.pop("sb_email", None)


def render_auth():
    client = get_supabase_client()

    access_token = st.session_state.get("sb_access_token")
    refresh_token = st.session_state.get("sb_refresh_token")

    if access_token and refresh_token:
        try:
            client.auth.set_session(access_token, refresh_token)
            response = client.auth.get_user()
            user = response.user

            if user:
                st.session_state["sb_user_id"] = user.id
                st.session_state["sb_email"] = user.email

                with st.sidebar:
                    st.write(f"Signed in: {user.email}")

                    if st.button("Log out", use_container_width=True):
                        try:
                            client.auth.sign_out()
                        finally:
                            clear_saved_session()
                        st.rerun()

                return client, user

        except Exception:
            clear_saved_session()

    st.title("📚 SmartDoc AI")
    st.subheader("Sign in or create your account")

    login_tab, signup_tab = st.tabs(
        ["Log in", "Create account"]
    )

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email address")
            password = st.text_input(
                "Password", type="password"
            )
            login_clicked = st.form_submit_button(
                "Log in", use_container_width=True
            )

        if login_clicked:
            if not email.strip() or not password:
                st.warning("Enter your email and password.")
            else:
                try:
                    result = client.auth.sign_in_with_password({
                        "email": email.strip(),
                        "password": password,
                    })

                    session = result.session

                    if session:
                        st.session_state["sb_access_token"] = (
                            session.access_token
                        )
                        st.session_state["sb_refresh_token"] = (
                            session.refresh_token
                        )
                        st.rerun()
                    else:
                        st.error("Login did not return a session.")

                except Exception:
                    st.error(
                        "Login failed. Check your details and "
                        "confirm your email if required."
                    )

    with signup_tab:
        with st.form("signup_form"):
            new_email = st.text_input(
                "Email address", key="signup_email"
            )
            new_password = st.text_input(
                "Create password",
                type="password",
                key="signup_password",
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password",
            )
            signup_clicked = st.form_submit_button(
                "Create account", use_container_width=True
            )

        if signup_clicked:
            if not new_email.strip():
                st.warning("Enter your email address.")
            elif len(new_password) < 8:
                st.warning(
                    "Use a password with at least 8 characters."
                )
            elif new_password != confirm_password:
                st.warning("The passwords do not match.")
            else:
                try:
                    result = client.auth.sign_up({
                        "email": new_email.strip(),
                        "password": new_password,
                    })

                    if result.session:
                        st.session_state["sb_access_token"] = (
                            result.session.access_token
                        )
                        st.session_state["sb_refresh_token"] = (
                            result.session.refresh_token
                        )
                        st.success("Account created.")
                        st.rerun()
                    else:
                        st.success(
                            "Account created! Check your email "
                            "to confirm it, then log in here."
                        )

                                except Exception as e:
                    st.error(f"Sign-up failed: {e}")
                    

    return client, None
  
