"""
src/auth_routes.py — Flask Blueprint for Authentication, Dashboard, and Admin

Routes
──────
  /auth/login          POST/GET   Login form + authentication
  /auth/register       POST/GET   Registration form (role always USER)
  /auth/logout         GET        Destroy session + redirect

  /dashboard           GET        User dashboard (submissions + activity)
  /dashboard/submissions GET      Full submission list
  /dashboard/activity    GET      Full activity list

  /admin               GET        Admin dashboard (stats + all data)
  /admin/toggle_user   POST       Enable/disable a user
  /admin/update_role   POST       Update a user's role
  /admin/delete_user   POST       Delete a user
  /admin/review_submission POST   Approve/reject a submission
  /admin/trigger_retrain   POST   Trigger retraining job
  /admin/api/activity  GET        JSON endpoint for activity filtering
  /admin/api/submission/<id>/samples GET JSON endpoint for submission samples

  /api/track_activity  POST       Client-side activity tracking

Security
────────
  • Every admin route checks session + role == ADMIN server-side.
  • Passwords are never stored in plaintext or in session/localStorage.
  • Sessions use Flask's built-in signed cookies (SECRET_KEY).
  • Registration always forces role = 'USER'.
"""

import os
import re
import uuid
import logging
import functools
from datetime import datetime, timezone

from flask import (
    Blueprint, render_template, request, redirect, url_for,
    session, flash, jsonify, abort,
)

from src.auth_models import UserManager

logger = logging.getLogger(__name__)

# ── Blueprint ─────────────────────────────────────────────────────────────────
auth_bp = Blueprint("auth", __name__)

# ── Shared user-manager instance (set from web_app.py at startup) ────────────
_user_mgr: UserManager | None = None


def init_auth(user_manager: UserManager):
    """Called once from web_app.py to inject the shared UserManager."""
    global _user_mgr
    _user_mgr = user_manager


def _mgr() -> UserManager:
    if _user_mgr is None:
        raise RuntimeError("UserManager not initialized — call init_auth() first.")
    return _user_mgr


# ═══════════════════════════════════════════════════════════════════════════════
# Decorators
# ═══════════════════════════════════════════════════════════════════════════════

def login_required(fn):
    """Redirect to login if the user is not authenticated."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "error")
            return redirect(url_for("auth.login_page"))
        return fn(*args, **kwargs)
    return wrapper


def admin_required(fn):
    """Return ACCESS DENIED if the user is not an ADMIN."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "error")
            return redirect(url_for("auth.login_page"))
        if session.get("role") != "ADMIN":
            return render_template("access_denied.html"), 403
        return fn(*args, **kwargs)
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# Auth Routes
# ═══════════════════════════════════════════════════════════════════════════════

@auth_bp.route("/auth/login", methods=["GET", "POST"])
def login_page():
    if request.method == "GET":
        # Already logged in → dashboard
        if "user_id" in session:
            return redirect(url_for("auth.dashboard"))
        return render_template("login.html")

    # POST — authenticate
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""

    if not username or not password:
        flash("Please enter your username/email and password.", "error")
        return render_template("login.html")

    user = _mgr().authenticate(username, password)
    if not user:
        # Intentionally vague to prevent enumeration
        flash("Invalid username/email or password.", "error")
        return render_template("login.html")

    # Build session
    sid = uuid.uuid4().hex
    session.clear()
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["full_name"] = user["full_name"]
    session["role"] = user["role"]
    session["session_id"] = sid
    session.permanent = True

    # Update DB
    _mgr().update_last_login(user["id"])
    _mgr().create_session(
        user["id"], sid,
        ip_address=request.remote_addr,
        user_agent=str(request.user_agent)[:256],
    )
    _mgr().log_activity(user["id"], "LOGIN", page="/auth/login", action="LOGIN_SUCCESS", session_id=sid)

    logger.info("User '%s' logged in (role=%s).", user["username"], user["role"])

    # Redirect by role
    if user["role"] == "ADMIN":
        return redirect(url_for("auth.admin_dashboard"))
    return redirect(url_for("auth.dashboard"))


@auth_bp.route("/auth/register", methods=["GET", "POST"])
def register_page():
    if request.method == "GET":
        if "user_id" in session:
            return redirect(url_for("auth.dashboard"))
        return render_template("register.html")

    # POST — create account
    full_name = (request.form.get("full_name") or "").strip()
    username = (request.form.get("username") or "").strip()
    email = (request.form.get("email") or "").strip()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm_password") or ""

    # Validation
    errors = []
    if not full_name:
        errors.append("Full name is required.")
    if not username or len(username) < 3:
        errors.append("Username must be at least 3 characters.")
    if not re.match(r'^[a-zA-Z0-9_]{3,30}$', username):
        errors.append("Username may only contain letters, numbers, and underscores.")
    if not email or "@" not in email:
        errors.append("A valid email is required.")
    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    if password != confirm:
        errors.append("Passwords do not match.")

    # Check existing
    if not errors:
        if _mgr().get_user_by_username(username):
            errors.append("That username is already taken.")
        if _mgr().get_user_by_email(email):
            errors.append("That email is already registered.")

    if errors:
        for e in errors:
            flash(e, "error")
        return render_template("register.html")

    # Create user — role is ALWAYS 'USER'. Never from form.
    user = _mgr().create_user(full_name, username, email, password, role="USER")
    if not user:
        flash("Registration failed. Please try a different username or email.", "error")
        return render_template("register.html")

    logger.info("New user registered: '%s' <%s>.", username, email)
    flash("Account created successfully! Please log in.", "success")
    return redirect(url_for("auth.login_page"))


@auth_bp.route("/auth/logout")
def logout():
    sid = session.get("session_id")
    uid = session.get("user_id")
    if uid and sid:
        _mgr().log_activity(uid, "LOGOUT", page="/auth/logout", action="LOGOUT", session_id=sid)
        _mgr().end_session(sid)
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("auth.login_page"))


# ═══════════════════════════════════════════════════════════════════════════════
# User Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

@auth_bp.route("/dashboard")
@login_required
def dashboard():
    uid = session["user_id"]
    user = _mgr().get_user_by_id(uid)
    submissions = _mgr().get_user_submissions(uid)
    activities = _mgr().get_user_activity(uid, limit=20)
    return render_template("dashboard.html", user=user, submissions=submissions, activities=activities)


@auth_bp.route("/dashboard/submissions")
@login_required
def my_submissions():
    uid = session["user_id"]
    user = _mgr().get_user_by_id(uid)
    submissions = _mgr().get_user_submissions(uid)
    return render_template("dashboard.html", user=user, submissions=submissions,
                           activities=_mgr().get_user_activity(uid, limit=20))


@auth_bp.route("/dashboard/activity")
@login_required
def my_activity():
    uid = session["user_id"]
    user = _mgr().get_user_by_id(uid)
    activities = _mgr().get_user_activity(uid, limit=100)
    submissions = _mgr().get_user_submissions(uid)
    return render_template("dashboard.html", user=user, submissions=submissions, activities=activities)


# ═══════════════════════════════════════════════════════════════════════════════
# Admin Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

@auth_bp.route("/admin")
@admin_required
def admin_dashboard():
    stats = _mgr().get_admin_stats()
    users = _mgr().get_all_users()
    submissions = _mgr().get_all_submissions()
    all_activity = _mgr().get_all_activity(limit=100)
    model_versions = _mgr().get_model_versions()
    training_jobs = _mgr().get_training_jobs()
    return render_template("admin_dashboard.html",
                           stats=stats, users=users, submissions=submissions,
                           all_activity=all_activity, model_versions=model_versions,
                           training_jobs=training_jobs)


@auth_bp.route("/admin/users")
@admin_required
def admin_users():
    return redirect(url_for("auth.admin_dashboard"))


@auth_bp.route("/admin/submissions")
@admin_required
def admin_submissions():
    return redirect(url_for("auth.admin_dashboard"))


@auth_bp.route("/admin/activity")
@admin_required
def admin_activity():
    return redirect(url_for("auth.admin_dashboard"))


@auth_bp.route("/admin/models")
@admin_required
def admin_models():
    return redirect(url_for("auth.admin_dashboard"))


@auth_bp.route("/admin/toggle_user", methods=["POST"])
@admin_required
def toggle_user():
    user_id = request.form.get("user_id", type=int)
    is_active = request.form.get("is_active") == "true"
    if user_id:
        _mgr().toggle_user_active(user_id, is_active)
        _mgr().log_activity(
            session["user_id"], "USER_MANAGED",
            page="/admin/users",
            action=f"{'ENABLED' if is_active else 'DISABLED'} user_id={user_id}",
            session_id=session.get("session_id"),
        )
    return redirect(url_for("auth.admin_dashboard"))


@auth_bp.route("/admin/review_submission", methods=["POST"])
@admin_required
def review_submission():
    sid = request.form.get("submission_id", type=int)
    status = request.form.get("status")
    reason = request.form.get("rejection_reason", "").strip() or None

    if sid and status in ("APPROVED", "REJECTED"):
        _mgr().review_submission(sid, status, reviewed_by=session["user_id"],
                                  rejection_reason=reason if status == "REJECTED" else None)
        _mgr().log_activity(
            session["user_id"], "SAMPLE_REVIEWED",
            page="/admin/submissions",
            action=f"{status} submission_id={sid}",
            session_id=session.get("session_id"),
        )
        flash(f"Submission #{sid} has been {status.lower()}.", "success")
    return redirect(url_for("auth.admin_dashboard"))


@auth_bp.route("/admin/update_role", methods=["POST"])
@admin_required
def update_role():
    user_id = request.form.get("user_id", type=int)
    role = request.form.get("role")
    if user_id and role in ("USER", "ADMIN"):
        # Prevent demoting the last admin if needed, but for simplicity:
        _mgr().update_user_role(user_id, role)
        _mgr().log_activity(
            session["user_id"], "USER_MANAGED",
            page="/admin/users",
            action=f"UPDATED ROLE to {role} user_id={user_id}",
            session_id=session.get("session_id"),
        )
        flash(f"User #{user_id} role updated to {role}.", "success")
    return redirect(url_for("auth.admin_dashboard"))


@auth_bp.route("/admin/delete_user", methods=["POST"])
@admin_required
def delete_user():
    user_id = request.form.get("user_id", type=int)
    if user_id:
        if user_id == session["user_id"]:
            flash("You cannot delete yourself.", "error")
        else:
            _mgr().delete_user(user_id)
            _mgr().log_activity(
                session["user_id"], "USER_MANAGED",
                page="/admin/users",
                action=f"DELETED user_id={user_id}",
                session_id=session.get("session_id"),
            )
            flash(f"User #{user_id} deleted successfully.", "success")
    return redirect(url_for("auth.admin_dashboard"))


@auth_bp.route("/admin/trigger_retrain", methods=["POST"])
@admin_required
def trigger_retrain():
    job_type = request.form.get("job_type", "Word Model Retraining")
    job_id = _mgr().create_training_job(job_type, session["user_id"])
    if job_id:
        _mgr().log_activity(
            session["user_id"], "RETRAINING_TRIGGERED",
            page="/admin/models",
            action=f"TRIGGERED JOB {job_type}",
            session_id=session.get("session_id"),
        )
        
        # Start background retraining thread
        from src.cloud_sync import start_cloud_retrain_thread
        start_cloud_retrain_thread(job_id)
        
        flash(f"Training job #{job_id} ({job_type}) has been started in the background.", "success")
    else:
        flash("Failed to create training job.", "error")
    return redirect(url_for("auth.admin_dashboard"))


@auth_bp.route("/admin/api/submission/<int:submission_id>/samples")
@admin_required
def admin_api_submission_samples(submission_id):
    """JSON endpoint for fetching samples of a submission."""
    samples = _mgr().get_submission_samples(submission_id)
    return jsonify([{
        "id": s["id"],
        "sample_path": s["sample_path"],
        "sample_type": s["sample_type"],
        "frame_count": s["frame_count"],
        "created_at": s["created_at"].strftime("%d-%m-%Y %H:%M") if s.get("created_at") else ""
    } for s in samples])


@auth_bp.route("/admin/api/activity")
@admin_required
def admin_api_activity():
    """JSON endpoint for admin activity filtering."""
    user_id = request.args.get("user_id", type=int)
    activity_type = request.args.get("activity_type")
    limit = request.args.get("limit", 200, type=int)

    rows = _mgr().get_all_activity(
        limit=limit, user_id=user_id, activity_type=activity_type,
    )
    result = []
    for r in rows:
        result.append({
            "id": r["id"],
            "user_id": r["user_id"],
            "full_name": r.get("full_name", ""),
            "username": r.get("username", ""),
            "activity_type": r["activity_type"],
            "page": r.get("page"),
            "action": r.get("action"),
            "time": r["created_at"].strftime("%H:%M") if r.get("created_at") else "",
            "date": r["created_at"].strftime("%d-%m-%Y") if r.get("created_at") else "",
        })
    return jsonify(result)


# ═══════════════════════════════════════════════════════════════════════════════
# Activity Tracking API (for client-side JS)
# ═══════════════════════════════════════════════════════════════════════════════

@auth_bp.route("/api/track_activity", methods=["POST"])
def track_activity():
    """Track a meaningful application action. Only works for logged-in users."""
    if "user_id" not in session:
        return jsonify({"ok": False}), 401

    data = request.get_json(silent=True) or {}
    activity_type = data.get("activity_type", "PAGE_VIEW")
    page = data.get("page")
    action = data.get("action")

    _mgr().log_activity(
        session["user_id"],
        activity_type=activity_type,
        page=page,
        action=action,
        session_id=session.get("session_id"),
    )
    # Touch the session
    sid = session.get("session_id")
    if sid:
        _mgr().touch_session(sid)

    return jsonify({"ok": True})
