#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base GUI components and utilities for Blackjack applications
"""

import tkinter as tk
from tkinter import ttk, messagebox
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from config import UIConfig, AppConfig

class BaseGUI(ABC):
    """Base class for GUI applications with common functionality"""
    
    def __init__(self, root: tk.Tk, title: str = "Blackjack Application"):
        self.root = root
        self.style = None
        self._setup_window(title)
        self._configure_styles()
    
    def _setup_window(self, title: str) -> None:
        """Setup main window properties"""
        self.root.title(title)
        self.root.geometry(f"{UIConfig.Dimensions.WINDOW_WIDTH}x{UIConfig.Dimensions.WINDOW_HEIGHT}")
        self.root.configure(bg=UIConfig.Colors.BACKGROUND_DARK)
        self.root.resizable(False, False)
    
    def _configure_styles(self) -> None:
        """Configure ttk styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure custom styles
        style_configs = {
            'Title.TLabel': {
                'font': UIConfig.Fonts.TITLE,
                'background': UIConfig.Colors.BACKGROUND_DARK,
                'foreground': UIConfig.Colors.WHITE
            },
            'Header.TLabel': {
                'font': UIConfig.Fonts.HEADER,
                'background': UIConfig.Colors.BACKGROUND_DARK,
                'foreground': UIConfig.Colors.WHITE
            },
            'Info.TLabel': {
                'font': UIConfig.Fonts.NORMAL,
                'background': UIConfig.Colors.BACKGROUND_DARK,
                'foreground': UIConfig.Colors.WHITE
            },
            'Success.TLabel': {
                'font': UIConfig.Fonts.HEADER,
                'background': UIConfig.Colors.BACKGROUND_DARK,
                'foreground': UIConfig.Colors.SUCCESS
            },
            'Danger.TLabel': {
                'font': UIConfig.Fonts.HEADER,
                'background': UIConfig.Colors.BACKGROUND_DARK,
                'foreground': UIConfig.Colors.DANGER
            }
        }
        
        for style_name, config in style_configs.items():
            self.style.configure(style_name, **config)
    
    def create_label_frame(self, parent: tk.Widget, text: str, 
                          width: Optional[int] = None, height: Optional[int] = None) -> tk.LabelFrame:
        """Create a styled label frame"""
        frame = tk.LabelFrame(
            parent, 
            text=text,
            bg=UIConfig.Colors.BACKGROUND_MEDIUM,
            fg=UIConfig.Colors.WHITE,
            font=UIConfig.Fonts.HEADER
        )
        
        if width and height:
            frame.configure(width=width, height=height)
            
        return frame
    
    def create_button(self, parent: tk.Widget, text: str, command: Callable,
                     bg_color: str = UIConfig.Colors.SUCCESS, 
                     fg_color: str = UIConfig.Colors.WHITE,
                     height: int = UIConfig.Dimensions.BUTTON_HEIGHT) -> tk.Button:
        """Create a styled button"""
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg_color,
            fg=fg_color,
            font=UIConfig.Fonts.BUTTON if height == 1 else UIConfig.Fonts.LARGE_BUTTON,
            relief='raised',
            bd=3 if height > 1 else 2,
            cursor='hand2',
            height=height
        )
    
    def create_entry(self, parent: tk.Widget, width: int = UIConfig.Dimensions.ENTRY_WIDTH) -> tk.Entry:
        """Create a styled entry widget"""
        return tk.Entry(
            parent,
            font=UIConfig.Fonts.HEADER,
            width=width
        )
    
    def create_combobox(self, parent: tk.Widget, values: list, 
                       width: int = UIConfig.Dimensions.COMBO_WIDTH) -> ttk.Combobox:
        """Create a styled combobox widget"""
        return ttk.Combobox(
            parent,
            values=values,
            state="readonly",
            width=width
        )
    
    def create_label(self, parent: tk.Widget, text: str,
                    bg_color: str = UIConfig.Colors.BACKGROUND_MEDIUM,
                    fg_color: str = UIConfig.Colors.WHITE,
                    font: tuple = UIConfig.Fonts.NORMAL,
                    **kwargs) -> tk.Label:
        """Create a styled label"""
        return tk.Label(
            parent,
            text=text,
            bg=bg_color,
            fg=fg_color,
            font=font,
            **kwargs
        )
    
    def show_error(self, message: str, title: str = "Error") -> None:
        """Show error message dialog"""
        messagebox.showerror(title, message)
    
    def show_info(self, message: str, title: str = "Information") -> None:
        """Show information message dialog"""
        messagebox.showinfo(title, message)
    
    def show_warning(self, message: str, title: str = "Warning") -> None:
        """Show warning message dialog"""
        messagebox.showwarning(title, message)
    
    def ask_yes_no(self, message: str, title: str = "Confirmation") -> bool:
        """Show yes/no confirmation dialog"""
        return messagebox.askyesno(title, message)
    
    @abstractmethod
    def setup_ui(self) -> None:
        """Abstract method to setup UI - must be implemented by subclasses"""
        pass

class GridLayoutMixin:
    """Mixin class for grid layout utilities"""
    
    @staticmethod
    def configure_grid_weights(widget: tk.Widget, rows: Optional[Dict[int, int]] = None, 
                             cols: Optional[Dict[int, int]] = None) -> None:
        """
        Configure grid row and column weights
        
        Args:
            widget: The widget to configure
            rows: Dictionary of {row_index: weight}
            cols: Dictionary of {column_index: weight}
        """
        if rows:
            for row, weight in rows.items():
                widget.grid_rowconfigure(row, weight=weight)
        
        if cols:
            for col, weight in cols.items():
                widget.grid_columnconfigure(col, weight=weight)
    
    @staticmethod
    def create_grid_frame(parent: tk.Widget, **grid_options) -> tk.Frame:
        """Create a frame and place it using grid"""
        frame = tk.Frame(parent)
        frame.grid(**grid_options)
        return frame

class ValidationMixin:
    """Mixin class for input validation utilities"""
    
    def validate_required_fields(self, fields: Dict[str, tk.Widget]) -> bool:
        """
        Validate that required fields are not empty
        
        Args:
            fields: Dictionary of {field_name: widget}
            
        Returns:
            True if all fields are valid, False otherwise
        """
        for field_name, widget in fields.items():
            if hasattr(widget, 'get') and not widget.get().strip():
                self.show_error(f"{field_name} is required!")
                widget.focus()
                return False
        return True
    
    def clear_fields(self, fields: Dict[str, tk.Widget]) -> None:
        """Clear all fields in the dictionary"""
        for widget in fields.values():
            if hasattr(widget, 'delete'):
                widget.delete(0, tk.END)
            elif hasattr(widget, 'set'):
                widget.set('')

class StatusBarMixin:
    """Mixin class for status bar functionality"""
    
    def create_status_bar(self, parent: tk.Widget) -> tk.Label:
        """Create a status bar at the bottom of the parent widget"""
        status_bar = self.create_label(
            parent,
            text="Ready",
            bg_color=UIConfig.Colors.BACKGROUND_DARK,
            fg_color=UIConfig.Colors.WHITE,
            font=UIConfig.Fonts.SMALL,
            relief=tk.SUNKEN,
            bd=1
        )
        return status_bar
    
    def update_status(self, status_bar: tk.Label, message: str, 
                     color: str = UIConfig.Colors.WHITE) -> None:
        """Update status bar message"""
        status_bar.config(text=message, fg=color)
        status_bar.update()

class ProgressMixin:
    """Mixin class for progress indication"""
    
    def create_progress_bar(self, parent: tk.Widget) -> ttk.Progressbar:
        """Create a progress bar widget"""
        progress = ttk.Progressbar(
            parent,
            mode='determinate',
            length=300
        )
        return progress
    
    def update_progress(self, progress_bar: ttk.Progressbar, value: int) -> None:
        """Update progress bar value"""
        progress_bar['value'] = value
        progress_bar.update()