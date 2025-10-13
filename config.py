#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration settings for Blackjack applications
"""

class UIConfig:
    """UI-related configuration constants"""
    
    # Colors
    class Colors:
        BACKGROUND_DARK = '#0f5132'
        BACKGROUND_MEDIUM = '#198754'
        SUCCESS = '#28a745'
        WARNING = '#ffc107'
        DANGER = '#dc3545'
        INFO = '#17a2b8'
        PRIMARY = '#6f42c1'
        SECONDARY = '#fd7e14'
        WHITE = 'white'
        BLACK = 'black'
        LIGHT_GRAY = 'lightgray'
    
    # Fonts
    class Fonts:
        TITLE = ('Arial', 16, 'bold')
        HEADER = ('Arial', 12, 'bold')
        NORMAL = ('Arial', 10)
        BUTTON = ('Arial', 10, 'bold')
        LARGE_BUTTON = ('Arial', 12, 'bold')
        SMALL = ('Arial', 8)
    
    # Dimensions
    class Dimensions:
        WINDOW_WIDTH = 800
        WINDOW_HEIGHT = 600
        PANEL_WIDTH = 380
        PANEL_HEIGHT = 300
        BUTTON_HEIGHT = 2
        SMALL_BUTTON_HEIGHT = 1
        RESULT_FRAME_HEIGHT = 80
        STAT_FRAME_HEIGHT = 30
        ENTRY_WIDTH = 25
        COMBO_WIDTH = 20
        TREEVIEW_HEIGHT = 8

class BlackjackConfig:
    """Blackjack game-related configuration"""
    
    # Card values and definitions
    CARD_VALUES = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
        'J': 10, 'Q': 10, 'K': 10, 'A': 11
    }
    
    CARD_OPTIONS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    
    # Game constants
    BLACKJACK_VALUE = 21
    DEALER_STAND_VALUE = 17
    DEFAULT_DECKS = 6
    DEFAULT_SIMULATIONS = 10000
    ACE_HIGH_VALUE = 11
    ACE_LOW_VALUE = 1
    
    # Basic Strategy Constants
    class BasicStrategy:
        PAIR_SPLIT_ACES = 11
        PAIR_SPLIT_EIGHTS = 8
        SOFT_HAND_STAND_THRESHOLD = 19
        SOFT_EIGHTEEN = 18
        HARD_HAND_STAND_THRESHOLD = 17
        HARD_HAND_DEALER_BUST_THRESHOLD = 6

class AppConfig:
    """Application-wide configuration"""
    
    # File settings
    DATA_FILE_PREFIX = "blackjack_data_"
    DATA_FILE_EXTENSION = ".json"
    
    # Default values
    DEFAULT_SIMULATION_COUNT = 10000
    DEFAULT_DECK_COUNT = 6
    
    # Validation
    MIN_SIMULATION_COUNT = 100
    MAX_SIMULATION_COUNT = 1000000
    MIN_DECK_COUNT = 1
    MAX_DECK_COUNT = 8
    
    # Messages
    class Messages:
        EMPTY_FIELDS_ERROR = "Please fill all fields!"
        INVALID_CARD_ERROR = "Invalid card: {}. Valid cards: 2-10, J, Q, K, A"
        SIMULATION_SUCCESS = "Simulation completed successfully"
        DATA_SAVED = "Data saved to '{}' successfully!"
        DATA_LOADED = "Data loaded from '{}' successfully!"
        CLEAR_CONFIRMATION = "Are you sure you want to clear all game history?"
        NO_DATA_WARNING = "No game data available yet!"
        
    # UI Text
    class UIText:
        APP_TITLE = "🃏 Blackjack Winning Percentage Analysis"
        NEW_GAME_SECTION = "NEW GAME"
        STATISTICS_SECTION = "STATISTICS"
        GAME_HISTORY_SECTION = "GAME HISTORY"
        PLAYER_CARDS_LABEL = "Player Cards:"
        DEALER_CARDS_LABEL = "Dealer Cards:"
        EXAMPLE_TEXT = "Example: A,K or 10,7,4"
        ADD_GAME_BUTTON = "🎯 ADD GAME"
        CLEAR_BUTTON = "🗑️ CLEAR"
        DETAILED_STATS_BUTTON = "📋 DETAILED STATISTICS"
        SAVE_DATA_BUTTON = "💾 SAVE DATA"
        LOAD_DATA_BUTTON = "📂 LOAD DATA"
        CLEAR_HISTORY_BUTTON = "🗑️ CLEAR HISTORY"