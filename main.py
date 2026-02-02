# main.py

import sys
import os

# ---- Torch DLL load fix for PyInstaller ----
if hasattr(sys, "_MEIPASS"):
    torch_lib = os.path.join(sys._MEIPASS, "_internal", "torch", "lib")
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]

# Hard-disable CUDA (safety)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import shutil
import re
import io
import contextlib

# from PyQt5.QtCore import QProcess

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QFrame,
    QScrollArea, QGridLayout, QTextEdit, QRadioButton, 
    QStackedWidget, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from backend.file_loader import load_story_file 
from backend.json_to_csv import run_json_to_csv

# Card Container Component
# Acts like a reusable component, like in React.
class Card(QFrame):
    def __init__(self, title=None, tint="#F0EAFE"):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #E4DDF5;
                font-size: 18px;
            }
        """)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        if title:
            title_header = QLabel(title)
            title_header.setStyleSheet(f"""
                QLabel {{
                    background: {tint};
                    padding: 8px 14px;
                    font-weight: 600;
                }}
            """)
            self.layout.addWidget(title_header)


# File Icon Widget (comes out after uploading file/s)
class FileIcon(QWidget):
    def __init__(self, filename, filepath, on_click):
        super().__init__()
        self.filename = filename
        self.filepath = filepath
        self.on_click = on_click

        # Get file extension
        ext = os.path.splitext(filename)[1].lower()
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)

        # Icon frame with document symbol
        self.icon_frame = QFrame()
        self.icon_frame.setCursor(Qt.PointingHandCursor)
        self.icon_frame.setFixedSize(100, 120)
        self.icon_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(
                    x1: 0, y1: 0,
                    x2: 0, y2: 1,
                    stop: 0 #FAF8FE,
                    stop: 1 #F0EBFF
                );
                border: 2px solid #D7CFF1;
                border-radius: 8px;
            }
            QFrame:hover {
                background: qlineargradient(
                    x1: 0, y1: 0,
                    x2: 0, y2: 1,
                    stop: 0 #F1ECFF,
                    stop: 1 #E5DCFF
                );
                border: 2px solid #C4B3E8;
            }
        """)

        # Document icon with extension label
        icon_layout = QVBoxLayout(self.icon_frame)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon_layout.setSpacing(0)
        
        # Document icon (simple rectangle with folded corner)
        doc_icon = QLabel("📄")
        doc_icon.setAlignment(Qt.AlignCenter)
        doc_icon.setStyleSheet("""
            font-size: 48px;
            background: transparent;
            border: none;
        """)
        
        # File extension badge
        ext_label = QLabel(ext.upper().replace(".", ""))
        ext_label.setAlignment(Qt.AlignCenter)
        ext_label.setStyleSheet("""
            background: #9B7EDC;
            color: white;
            font-size: 10px;
            font-weight: bold;
            padding: 3px 8px;
            border-radius: 3px;
            border: none;
        """)
        
        icon_layout.addStretch()
        icon_layout.addWidget(doc_icon, alignment=Qt.AlignCenter)
        icon_layout.addSpacing(5)
        icon_layout.addWidget(ext_label, alignment=Qt.AlignCenter)
        icon_layout.addStretch()

        main_layout.addWidget(self.icon_frame, alignment=Qt.AlignCenter)

        # Filename label below the icon
        filename_label = QLabel(filename)
        filename_label.setWordWrap(True)
        filename_label.setAlignment(Qt.AlignCenter)
        filename_label.setMaximumWidth(110)
        filename_label.setStyleSheet("""
            font-size: 11px;
            color: #5A4A7A;
            background: transparent;
            border: none;
            padding: 0px;
        """)

        main_layout.addWidget(filename_label, alignment=Qt.AlignCenter)

    def mousePressEvent(self, event):
        self.on_click(self)

    def set_selected(self, selected):
        if selected:
            self.icon_frame.setStyleSheet("""
                QFrame {
                    background: qlineargradient(
                        x1: 0, y1: 0,
                        x2: 0, y2: 1,
                        stop: 0 #E8DFFF,
                        stop: 1 #D5C8F5
                    );
                    border: 3px solid #9B7EDC;
                    border-radius: 8px;
                }
                QFrame:hover {
                    background: qlineargradient(
                        x1: 0, y1: 0,
                        x2: 0, y2: 1,
                        stop: 0 #E8DFFF,
                        stop: 1 #D5C8F5
                    );
                    border: 3px solid #9B7EDC;
                }
            """)
        else:
            self.icon_frame.setStyleSheet("""
                QFrame {
                    background: qlineargradient(
                        x1: 0, y1: 0,
                        x2: 0, y2: 1,
                        stop: 0 #FAF8FE,
                        stop: 1 #F0EBFF
                    );
                    border: 2px solid #D7CFF1;
                    border-radius: 8px;
                }
                QFrame:hover {
                    background: qlineargradient(
                        x1: 0, y1: 0,
                        x2: 0, y2: 1,
                        stop: 0 #F1ECFF,
                        stop: 1 #E5DCFF
                    );
                    border: 2px solid #C4B3E8;
                }
            """)


# Main Application Window
class FileUploadUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MemoWeave")
        self.setMinimumSize(1200, 650)

        self.uploaded_files = []
        self.currently_selected_file_icon = None
        self.file_grid_max_columns = 5

        # Analysis / pipeline state
        self.pipeline_running = False
        self.last_executed_rule = None

        self.init_ui()
        self.set_controls_enabled(False)

    def set_controls_enabled(self, enabled):
        self.select_file_button.setEnabled(enabled)
        self.delete_file_button.setEnabled(enabled)
        self.analyze_button.setEnabled(enabled)
        for btn in self.rule_buttons.values():
            btn.setEnabled(enabled)

    # UI Setup (Most-parent container)
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(20)

        left_panel_layout = QVBoxLayout()
        right_panel_layout = QVBoxLayout()

        main_layout.addLayout(left_panel_layout, 3) 
        main_layout.addLayout(right_panel_layout, 2) 

        # RAW STORY CARD
        raw_story_card = Card("Raw Story", "#F3DDF7")

        upload_buttons_layout = QHBoxLayout()
        upload_buttons_layout.setContentsMargins(10, 10, 10, 10)
        upload_buttons_layout.setSpacing(10)
        
        self.upload_files_button = QPushButton("Upload Files")
        self.upload_files_button.clicked.connect(self.upload_files)

        self.select_file_button = QPushButton("Select File")
        self.select_file_button.setEnabled(False)
        self.select_file_button.clicked.connect(self.load_selected_file)

        self.delete_file_button = QPushButton("Delete File")
        self.delete_file_button.setEnabled(False)
        self.delete_file_button.clicked.connect(self.delete_selected_file)

        upload_buttons_layout.addWidget(self.upload_files_button)
        upload_buttons_layout.addWidget(self.select_file_button)
        upload_buttons_layout.addWidget(self.delete_file_button)

        raw_story_card.layout.addLayout(upload_buttons_layout)

        # System Feedback
        self.system_feedback = QLabel("")
        self.system_feedback.setAlignment(Qt.AlignCenter)
        self.system_feedback.setVisible(False)
        self.system_feedback.setStyleSheet("""
            background-color: #F2F0FF;
            border: 1px solid #CFC7EE;
            border-radius: 6px;
            padding: 6px;
            font-size: 12px;
            color: #4A3F7A;
        """)

        left_panel_layout.addWidget(self.system_feedback)

        # Stacked widget switches between file grid view and text view
        self.content_stack = QStackedWidget()
        self.content_stack.setMinimumHeight(230)
        self.content_stack.setStyleSheet("""
            QStackedWidget {
                border: none;
                padding-bottom: 12px;
            }
        """)

        # File grid view
        self.file_grid_scroll = QScrollArea()
        self.file_grid_scroll.setWidgetResizable(True)
        self.file_grid_scroll.setStyleSheet("""
            QScrollArea {
                background: #FAF8FE;
                border: 2px dashed #D7CFF1;
                margin: 0px 12px;
            }
        """)

        self.file_grid_container = QWidget()
        self.file_grid_layout = QGridLayout(self.file_grid_container)
        self.file_grid_layout.setSpacing(12)
        self.file_grid_scroll.setWidget(self.file_grid_container)

        # Upload prompt (will be hidden when files are added)
        self.upload_hint_label = QLabel("Drag and drop your story file/s here\n(.txt files)")
        self.upload_hint_label.setAlignment(Qt.AlignCenter)
        self.upload_hint_label.setStyleSheet("color: #9A90B8; background-color: transparent; border:none;")
        self.file_grid_layout.addWidget(self.upload_hint_label, 0, 0, 1, self.file_grid_max_columns)

        self.content_stack.addWidget(self.file_grid_scroll)

        # Text view for displaying file content
        self.file_content_text_view = QTextEdit()
        self.file_content_text_view.setReadOnly(True)
        self.content_stack.addWidget(self.file_content_text_view)

        raw_story_card.layout.addWidget(self.content_stack, stretch=1)

        # Mode switch radio buttons
        mode_switch_layout = QHBoxLayout()
        mode_switch_layout.setSpacing(15)
        mode_switch_layout.setContentsMargins(0, 0, 0, 16)
        mode_switch_layout.addStretch()

        self.file_view_mode_radio = QRadioButton("File Mode")
        self.file_view_mode_radio.setChecked(True)
        self.text_view_mode_radio = QRadioButton("Text Mode")
        self.text_view_mode_radio.setEnabled(False)

        self.file_view_mode_radio.toggled.connect(self.switch_view_mode)
        self.text_view_mode_radio.toggled.connect(self.switch_view_mode)

        mode_switch_layout.addWidget(self.file_view_mode_radio)
        mode_switch_layout.addWidget(self.text_view_mode_radio)
        mode_switch_layout.addStretch()

        raw_story_card.layout.addLayout(mode_switch_layout)
        left_panel_layout.addWidget(raw_story_card, stretch=2)

        # -----------------------------
        # Story Consistency Rules Card
        # -----------------------------
        story_rules_card = Card("Story Consistency Rules", "#DDE8FF")

        rules_container = QWidget()
        rules_layout = QGridLayout(rules_container)
        rules_layout.setSpacing(14)

        # Make buttons responsive
        for i in range(2):
            rules_layout.setColumnStretch(i, 1)

        self.selected_rule = None
        self.rule_buttons = {}

        def create_rule_button(title, description, key):
            btn = QPushButton(title)
            btn.setCheckable(True)
            btn.setEnabled(False)

            btn.setStyleSheet("""
                QPushButton {
                    background: white;
                    border: 2px solid #CFC7EE;
                    border-radius: 10px;
                    padding: 12px;
                    font-weight: 600;
                }
                QPushButton:checked {
                    background: #EEE8FF;
                    border: 3px solid #7D5FB5;
                }
            """)

            btn.clicked.connect(lambda: self.select_rule(key))

            desc = QLabel(description)
            desc.setWordWrap(True)
            desc.setAlignment(Qt.AlignCenter)
            desc.setStyleSheet("font-size: 15px; color: #4E4A63;")

            wrapper = QWidget()
            v = QVBoxLayout(wrapper)
            v.addWidget(btn)
            v.addWidget(desc)

            self.rule_buttons[key] = btn
            return wrapper

        # Add only the remaining two rules
        rules_layout.addWidget(
            create_rule_button(
                "Temporal Consistency ⏱️",
                "Checks that events happen in a logical order and at the right times. Detects contradictions in timing or overlapping events.",
                "temporal"
            ), 0, 0
        )

        rules_layout.addWidget(
            create_rule_button(
                "Role Completeness 🎭",
                "Checks that all important characters and tools are present when something happens in the story.",
                "role_completeness"
            ), 0, 1
        )

        story_rules_card.layout.addWidget(rules_container)
        left_panel_layout.addWidget(story_rules_card)


        # Inconsistencies Card 
        inconsistencies_card = Card("Memo Weave Feedback", "#FFE2E2")
        self.inconsistencies_output = QTextEdit()
        self.inconsistencies_output.setReadOnly(True)
        self.inconsistencies_output.setPlaceholderText(
            "Detail inconsistencies, story-rule conflicts, and other violations flagged will appear here..."
        )
        inconsistencies_card.layout.addWidget(self.inconsistencies_output)
        right_panel_layout.addWidget(inconsistencies_card)

        # Progress Messaging Card (takes what used to be outputted in the terminal and hoists it to UI) 
        progress_card = Card("Memo Weave System Progress", "#EFEFEF")
        self.progress_output = QTextEdit()
        self.progress_output.setReadOnly(True)

        # Placeholder only. Real messages vary.
        self.progress_output.setPlaceholderText(
            "Reading Text...\nSegmenting Chapters...\nTokenizing Sentences...\nAnnotating Linguistics...\nLoading Sentences...\nConstructing Event Frames...\nFilling gaps...\nExtracting Time Expressions..."
        )
        progress_card.layout.addWidget(self.progress_output)
        right_panel_layout.addWidget(progress_card)

        # ANALYZE BUTTON 
        self.analyze_button = QPushButton("✨ Analyze by Memo Weave")
        self.analyze_button.setEnabled(False)
        self.analyze_button.setFixedHeight(50)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                 background: qlineargradient(
                    x1: 0, y1: 0,
                    x2: 1, y2: 0,
                    stop: 0 rgba(125, 95, 181, 191),
                    stop: 1 rgba(199, 132, 255, 255)
                );
                font-size: 18px;
                font-weight: 600;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #8A6AD1;
                color: white;
            }
        """)
        right_panel_layout.addWidget(self.analyze_button, alignment=Qt.AlignRight)
        self.analyze_button.clicked.connect(self.run_pipeline)

    def select_rule(self, selected_key):
        # Prevent changing rules during pipeline
        if self.pipeline_running:
            self.show_system_feedback("Analysis in progress. Please wait.")
            return

        if self.selected_rule == selected_key:
            return

        # --- RESET FEEDBACK ---
        self.reset_feedback_output()

        # Set button checked state
        for key, btn in self.rule_buttons.items():
            btn.setChecked(key == selected_key)

        # Update selected rule
        self.selected_rule = selected_key
        # Enable Analyze button if conditions met
        self.update_analyze_state()

    def memory_has_events(self, memory_path):
        """
        Returns True only if memory_module.json exists
        AND contains at least one extracted event.
        """
        try:
            with open(memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            events = data.get("events")
            return isinstance(events, list) and len(events) > 0

        except Exception:
            return False

    # Pipeline-reaching helper functions
    def run_pipeline(self):
        # Prevent double execution
        if self.pipeline_running:
            return

        # Must have a file
        if not self.currently_selected_file_icon:
            self.progress_output.setPlainText("No file selected.")
            return

        # Must have a rule selected
        if not self.selected_rule:
            self.inconsistencies_output.setPlainText(
                "Please select a story consistency rule."
            )
            return

        memory_path = "output/memory/memory_module.json"

        # Reuse memory ONLY if it exists AND contains events
        if os.path.exists(memory_path) and self.memory_has_events(memory_path):
            self.progress_output.append(
                f"Using existing memory:\n{memory_path}"
            )
            QApplication.processEvents()
            self.run_rule_checker()
            return
        elif os.path.exists(memory_path):
            self.progress_output.append(
                "[WARN] Memory module exists but contains no events. Rebuilding..."
            )
            QApplication.processEvents()

        # --- RUN PIPELINE ---
        self.pipeline_running = True
        self.analyze_button.setEnabled(False)

        input_file = self.currently_selected_file_icon.filepath

        self.progress_output.clear()
        self.progress_output.append(
            f"Running Temporal Memory Pipeline on:\n{input_file}\n"
        )
        QApplication.processEvents()

        # 🔥 Capture print() output from pipeline
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                from backend.pipeline import run_pipeline as backend_run_pipeline
                backend_run_pipeline(input_file)

            # Flush captured stdout
            for line in stdout_buffer.getvalue().splitlines():
                self.append_progress(line)

            # Flush captured stderr
            for line in stderr_buffer.getvalue().splitlines():
                self.append_progress(f"[WARN] {line}")

            self.pipeline_finished()

        except Exception as e:
            self.append_progress(f"[ERROR] Pipeline failed: {e}")
            self.pipeline_running = False
            self.analyze_button.setEnabled(True)

    def run_rule_checker(self):
        memory_path = "output/memory/memory_module.json"
        if not os.path.exists(memory_path):
            self.inconsistencies_output.setPlainText(
                "No memory module found. Please click Analyze to build story memory first."
            )
            self.analyze_button.setEnabled(True)
            return

        if not self.selected_rule:
            self.inconsistencies_output.setPlainText(
                "Error: Please select exactly one story consistency rule."
            )
            self.analyze_button.setEnabled(True)
            return

        self.progress_output.append("\nPreparing memory for reasoning...")
        QApplication.processEvents()

        # STEP 1: JSON → CSV conversion
        csv_file_map = {
            "temporal": "output/memory/temporal_consistency.csv",
            "role_completeness": "output/memory/role_completeness.csv"
        }
        csv_path = csv_file_map.get(self.selected_rule)

        if (
            not os.path.exists(csv_path)
            or os.path.getmtime(csv_path) < os.path.getmtime(memory_path)
            or self.last_executed_rule != self.selected_rule
        ):
            try:
                run_json_to_csv(
                    memory_module_path=memory_path,
                    rule_class=self.selected_rule,
                    log_callback=self.progress_output.append
                )
                self.progress_output.append("Memory projection complete.")
            except Exception as e:
                self.progress_output.append(f"[ERROR] JSON → CSV failed: {e}")
                self.analyze_button.setEnabled(True)
                return
        else:
            self.progress_output.append("Reusing existing CSV projection.")
        QApplication.processEvents()

        # STEP 2: LLM reasoning & human-readable feedback
        try:
            self.progress_output.append("\nSending story data to AI for reasoning...")
            QApplication.processEvents()

            if self.selected_rule == "temporal":
                from backend.events import generate_feedback as generate_feedback_func
            elif self.selected_rule == "role_completeness":
                from backend.character import generate_feedback as generate_feedback_func

            feedback = generate_feedback_func(csv_path)
            self.inconsistencies_output.setHtml(self.format_feedback_html(feedback))
            self.progress_output.append("\nNarrative Validation Complete!!")
        except Exception as e:
            self.inconsistencies_output.setPlainText(f"[ERROR] LLM feedback generation failed: {e}")

        # Re-enable Analyze button
        self.analyze_button.setEnabled(True)
        self.last_executed_rule = self.selected_rule

    def format_feedback_html(self, feedback_text):
        """
        Converts **bold** markers and text inside quotes "like this" into HTML <b> tags
        for QTextEdit.
        """
        # Replace **text** with <b>text</b>
        html_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", feedback_text)
        
        # Replace "text in quotes" with <b>text in quotes</b>
        html_text = re.sub(r'"(.*?)"', r"<b>\1</b>", html_text)
        
        # Replace newlines with <br> for HTML display
        html_text = html_text.replace("\n", "<br>")
        return html_text

    def pipeline_finished(self):
        self.pipeline_running = False
        self.progress_output.append("\nFinished reading your story...")

        memory_path = "output/memory/memory_module.json"
        if self.memory_has_events(memory_path):
            self.run_rule_checker()
        else:
            self.progress_output.append(
                "[ERROR] No events extracted. Rule checking aborted."
            )
            self.analyze_button.setEnabled(True)

    # File Upload and Selection Logic
    def upload_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Files")
        for path in file_paths:
            filename = os.path.basename(path)
            if any(icon.filepath == path for icon in self.uploaded_files):
                self.show_system_feedback("This file is already uploaded.")
                continue
            self.add_file_icon(filename, path)

    def add_file_icon(self, filename, filepath):
        # Hide the upload hint label when first file is added
        if len(self.uploaded_files) == 0:
            self.upload_hint_label.hide()

        file_index = len(self.uploaded_files)
        grid_row = file_index // self.file_grid_max_columns
        grid_col = file_index % self.file_grid_max_columns

        file_icon = FileIcon(filename, filepath, self.select_file_icon)
        self.file_grid_layout.addWidget(file_icon, grid_row, grid_col, alignment=Qt.AlignTop | Qt.AlignHCenter)
        self.uploaded_files.append(file_icon)

    def select_file_icon(self, file_icon):
        # Reset output folders and feedback for new file
        self.reset_output_dirs()
        self.reset_feedback_output()

        if self.currently_selected_file_icon:
            self.currently_selected_file_icon.set_selected(False)

        self.currently_selected_file_icon = file_icon
        file_icon.set_selected(True)

        # RESET analysis state when a new file is selected
        self.last_executed_rule = None
        self.selected_rule = None

        # Disable rule buttons & Analyze button
        for btn in self.rule_buttons.values():
            btn.setEnabled(True)  # enabled after file selected
            btn.setChecked(False)
        self.analyze_button.setEnabled(False)

        # ENABLE CONTROLS
        self.set_controls_enabled(True)
        self.text_view_mode_radio.setEnabled(True)
        self.select_file_button.setEnabled(True)

        # CLEAR OUTPUTS
        self.inconsistencies_output.clear()
        self.progress_output.clear()

    def load_selected_file(self):
        if not self.currently_selected_file_icon:
            return
        
        try:
            # Try UTF-8 encoding first
            try:
                with open(self.currently_selected_file_icon.filepath, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except UnicodeDecodeError:
                # Fall back to latin-1 if UTF-8 fails
                with open(self.currently_selected_file_icon.filepath, "r", encoding="latin-1") as f:
                    file_content = f.read()

            self.file_content_text_view.setPlainText(file_content)
            self.text_view_mode_radio.setChecked(True)
        except Exception as e:
            self.file_content_text_view.setPlainText(f"Error loading file: {str(e)}")

    # Delete Uploaded Files
    def delete_selected_file(self):
        if not self.currently_selected_file_icon:
            return
        
        reply = QMessageBox.question(
            self,
            "Delete File",
            f"Are you sure you want to delete:\n\n{self.currently_selected_file_icon.filename}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        icon = self.currently_selected_file_icon
        self.file_grid_layout.removeWidget(icon)
        icon.deleteLater()

        self.uploaded_files.remove(icon)
        self.refresh_file_grid()
        self.currently_selected_file_icon = None

        if not self.uploaded_files:
            self.upload_hint_label.show()

        self.set_controls_enabled(False)
        self.text_view_mode_radio.setEnabled(False)
        self.file_view_mode_radio.setChecked(True)

        self.inconsistencies_output.clear()
        self.progress_output.clear()

        self.show_system_feedback("File removed.")
        self.update_analyze_state()

    def refresh_file_grid(self):
        # Remove all widgets from the grid (except the upload hint)
        for i in reversed(range(self.file_grid_layout.count())):
            widget = self.file_grid_layout.itemAt(i).widget()
            if widget and widget is not self.upload_hint_label:
                self.file_grid_layout.removeWidget(widget)

        # Re-add file icons in order
        for index, icon in enumerate(self.uploaded_files):
            row = index // self.file_grid_max_columns
            col = index % self.file_grid_max_columns
            self.file_grid_layout.addWidget(
                icon,
                row,
                col,
                alignment=Qt.AlignTop | Qt.AlignHCenter
            )

    # 
    def show_system_feedback(self, message, duration=2500):
            self.system_feedback.setText(message)
            self.system_feedback.setVisible(True)
            QTimer.singleShot(duration, lambda: self.system_feedback.setVisible(False))

    def switch_view_mode(self):
        if self.file_view_mode_radio.isChecked():
            self.content_stack.setCurrentIndex(0)
            self.upload_files_button.setEnabled(True)
            self.select_file_button.setEnabled(self.currently_selected_file_icon is not None)
            self.delete_file_button.setEnabled(self.currently_selected_file_icon is not None)
        elif self.text_view_mode_radio.isChecked():
            self.content_stack.setCurrentIndex(1)
            self.upload_files_button.setEnabled(False)
            self.select_file_button.setEnabled(False)

        self.update_analyze_state()

    def update_analyze_state(self):
        can_analyze = (
            self.selected_rule is not None
            and not self.pipeline_running
            and (
                self.text_view_mode_radio.isChecked()
                or self.currently_selected_file_icon is not None
            )
        )

        self.analyze_button.setEnabled(can_analyze)

    def reset_output_dirs(self):
        """Delete existing output/memory and output/preprocessed folders."""
        import shutil
        output_dirs = [
            "output/memory",
            "output/preprocessed"
        ]
        for d in output_dirs:
            if os.path.exists(d):
                shutil.rmtree(d)

    def reset_feedback_output(self):
        """Reset Memo Weave Feedback to placeholder text."""
        self.inconsistencies_output.clear()
        self.inconsistencies_output.setPlaceholderText(
            "Detail inconsistencies, story-rule conflicts, and other violations flagged will appear here..."
        )

    def append_progress(self, text):
        self.progress_output.append(text.rstrip())
        QApplication.processEvents()

# App Entry Point
if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QWidget {
            background-color: #F6F3FA;
            font-family: 'Segoe UI';
        }
        QTextEdit {
            padding: 10px;
            border: 1px solid #E0DAF0;
        }
        QPushButton {
            padding: 10px 18px;
        }
    """)

    font = QFont()
    font.setPointSize(11)
    app.setFont(font)

    window = FileUploadUI()
    window.show()

    sys.exit(app.exec_())