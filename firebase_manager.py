# utils/firebase_logging.py

import uuid
import datetime as dt
import pytz
import os
import json

import firebase_admin
from firebase_admin import credentials, db, storage

HKT = pytz.timezone('Asia/Hong_Kong')

_LOGGER = None  # kept for backward compatibility, but we now prefer session_state

import streamlit as st

def initialize_firebase():
    # This supports TOML secrets with private_key as a one-line string with \n's
    if 'firebase' in st.secrets:
        svcdict = dict(st.secrets['firebase']['service_account'])
        bucket = st.secrets['firebase'].get('storage_bucket', f"{svcdict['project_id']}.appspot.com")
        dburl = st.secrets['firebase']['database_url']
        # Convert \n to real newlines if required
        if "\\n" in svcdict["private_key"]:
            svcdict["private_key"] = svcdict["private_key"].replace("\\n", "\n")
        if not firebase_admin._apps:
            cred = credentials.Certificate(svcdict)
            firebase_admin.initialize_app(
                cred, {
                    'databaseURL': dburl,
                    'storageBucket': bucket
                }
            )
        # Optionally return Firebase DB reference
        return db.reference("/")
    else:
        raise RuntimeError("No firebase config present in Streamlit secrets")


def _get_or_create_session_id(st):
    key = "_session_id"
    if key not in st.session_state:
        st.session_state[key] = uuid.uuid4().hex
    return st.session_state[key]


class FirebaseLogger:
    def __init__(self, st, run_context=None):

        self.st = st
        self.bucket = None      # Will be set if Firebase enabled
        self.events_ref = None  # Points to runs/<run_id>/events
        self.screens_ref = None # Points to runs/<run_id>/screens
        self.run_ref = None     # Points to this run's root
        self.run_id = None
        self.session_id = _get_or_create_session_id(st)

        if "firebase" in st.secrets:
            svc_dict = dict(st.secrets["firebase"]["service_account"])
            bucket = st.secrets.get("firebase", {}).get("storage_bucket") or f"{svc_dict['project_id']}.appspot.com"

            db_url = st.secrets["firebase"]["database_url"]  # Ensure you have this in secrets.toml

            # Initialize firebase only once per process
            if not firebase_admin._apps:
                cred = credentials.Certificate(svc_dict)
                app = firebase_admin.initialize_app(cred, {
                    "databaseURL": db_url,
                    "storageBucket": bucket,
                })
            else:
                app = firebase_admin.get_app()

            # Reference: logs/streamlit/<session_id>
            self.db = db.reference(f"logs/streamlit/{self.session_id}")

            # --- START OF RUN-ISOLATION PATCH ---
            # Create a new, sortable run_id: e.g. 20250912T161100_ab1c2d

            # Use current time in HKT for run_id (for local files/folders)
            now_hkt = dt.datetime.now(HKT)
            self.run_id = now_hkt.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
            self.local_log_events = []
            # Pointer to this run:
            self.run_ref = self.db.child("runs").child(self.run_id)
            self.events_ref = self.run_ref.child("events")
            self.screens_ref = self.run_ref.child("screens")
            # Mark start/meta fields
            meta = {
                "started_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            }
            if run_context:
                meta["context"] = run_context
            self.run_ref.update(meta)

            # Set up gs bucket as before
            self.bucket = storage.bucket(app=app)


    def info(self, msg, **extra):
        self._event("INFO", msg, extra)

    def warn(self, msg, **extra):
        self._event("WARN", msg, extra)

    def error(self, msg, **extra):
        self._event("ERROR", msg, extra)

    def _event(self, level, message, extra=None):
        # Only log to local, NOT Firebase DB
        payload = {
                    "ts": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                    "level": level,
                    "message": message
                }
        if extra:
            payload.update(extra)
        self.local_log_events.append(payload)
        # DO NOT: self.events_ref.push(payload)

    def upload_file_to_firebase(self, local_fp, remote_path):
        blob = self.bucket.blob(remote_path)
        blob.upload_from_filename(local_fp)
        return f"gs://{self.bucket.name}/{remote_path}"

    def export_log_json(self, log_dir):
        # Write file locally as before
        os.makedirs(log_dir, exist_ok=True)
        fp = os.path.join(log_dir, f"{self.run_id}_logs.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(self.local_log_events, f, ensure_ascii=False, indent=2)
        return fp

    def upload_scrcap_to_firebase(self, local_fp, name_hint="screenshot"):
        # Upload individual screenshot file
        run_screens_dir = f"runs/{self.session_id}/{self.run_id}/screens/"
        remote_path = run_screens_dir + os.path.basename(local_fp)
        return self.upload_file_to_firebase(local_fp, remote_path)

    def upload_json_to_firebase(self, json_fp):
        run_dir = f"runs/{self.session_id}/{self.run_id}/"
        remote_path = run_dir + os.path.basename(json_fp)
        return self.upload_file_to_firebase(json_fp, remote_path)


    def end_run(self, status="completed", summary=None):
        if self.run_ref:
            self.run_ref.update({
                "ended_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                 "status": status,
                 "summary": summary or {},
             })


def ensure_logger(st, run_context=None):
    # Prefer a per-session singleton stored in session_state
    if "fb_logger" not in st.session_state:
        st.session_state["fb_logger"] = FirebaseLogger(st, run_context=run_context)
    elif run_context and st.session_state["fb_logger"].run_ref:
        st.session_state["fb_logger"].run_ref.update({"context": run_context})
    return st.session_state["fb_logger"]


def get_logger(st):
    # Convenience accessor if needed elsewhere
    return st.session_state.get("fb_logger")


def patch_streamlit_logging(st):
    fb = ensure_logger(st)
    originals = {}
    for name in ["write", "info", "warning", "error", "success", "code"]:
        orig = getattr(st, name)
        originals[name] = orig

        def make_logged(fn, fname=name):
            def inner(*args, **kwargs):
                msg = " ".join(str(a) for a in args) if args else ""
                if fname in ["error", "warning"]:
                    fb.error(msg)
                else:
                    fb.info(msg)
                return fn(*args, **kwargs)
            return inner

        setattr(st, name, make_logged(orig))
    return originals
