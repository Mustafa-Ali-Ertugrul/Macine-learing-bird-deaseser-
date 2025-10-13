#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities for Blackjack applications including validation, error handling, and common functions
"""

import re
from typing import List, Tuple, Optional, Union
from config import BlackjackConfig, AppConfig

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class CardValidator:
    """Validator for card inputs and game logic"""
    
    @staticmethod
    def validate_card_string(card_str: str) -> str:
        """
        Validate and normalize a single card string
        
        Args:
            card_str: Card string to validate (e.g., 'A', 'K', '10')
            
        Returns:
            Normalized card string
            
        Raises:
            ValidationError: If card is invalid
        """
        card_str = card_str.upper().strip()
        
        if card_str not in BlackjackConfig.CARD_OPTIONS:
            raise ValidationError(
                AppConfig.Messages.INVALID_CARD_ERROR.format(card_str)
            )
        
        return card_str
    
    @staticmethod
    def validate_cards_input(cards_input: str) -> List[str]:
        """
        Validate and parse cards input string
        
        Args:
            cards_input: Comma-separated cards string (e.g., 'A,K,10')
            
        Returns:
            List of validated card strings
            
        Raises:
            ValidationError: If any card is invalid or input format is wrong
        """
        if not cards_input or not cards_input.strip():
            raise ValidationError(AppConfig.Messages.EMPTY_FIELDS_ERROR)
        
        # Split and clean cards
        cards = [card.strip() for card in cards_input.split(',')]
        
        # Remove empty strings
        cards = [card for card in cards if card]
        
        if not cards:
            raise ValidationError("No valid cards found in input")
        
        # Validate each card
        validated_cards = []
        for card in cards:
            validated_cards.append(CardValidator.validate_card_string(card))
        
        return validated_cards
    
    @staticmethod
    def validate_simulation_parameters(simulations: str, decks: str) -> Tuple[int, int]:
        """
        Validate simulation parameters
        
        Args:
            simulations: Number of simulations as string
            decks: Number of decks as string
            
        Returns:
            Tuple of (simulations, decks) as integers
            
        Raises:
            ValidationError: If parameters are invalid
        """
        try:
            sim_count = int(simulations)
            deck_count = int(decks)
        except ValueError:
            raise ValidationError("Simulation and deck counts must be integers")
        
        if not (AppConfig.MIN_SIMULATION_COUNT <= sim_count <= AppConfig.MAX_SIMULATION_COUNT):
            raise ValidationError(
                f"Simulation count must be between {AppConfig.MIN_SIMULATION_COUNT} "
                f"and {AppConfig.MAX_SIMULATION_COUNT}"
            )
        
        if not (AppConfig.MIN_DECK_COUNT <= deck_count <= AppConfig.MAX_DECK_COUNT):
            raise ValidationError(
                f"Deck count must be between {AppConfig.MIN_DECK_COUNT} "
                f"and {AppConfig.MAX_DECK_COUNT}"
            )
        
        return sim_count, deck_count

class CardCalculator:
    """Calculator for card values and hand totals"""
    
    @staticmethod
    def get_card_value(card_str: str) -> int:
        """
        Get numeric value of a card
        
        Args:
            card_str: Card string (e.g., 'A', 'K', '10')
            
        Returns:
            Numeric value of the card
        """
        normalized_card = CardValidator.validate_card_string(card_str)
        return BlackjackConfig.CARD_VALUES[normalized_card]
    
    @staticmethod
    def calculate_hand_total(cards: List[str]) -> int:
        """
        Calculate total value of a hand, handling Aces optimally
        
        Args:
            cards: List of card strings
            
        Returns:
            Optimal total value of the hand
        """
        total = 0
        aces = 0
        
        for card in cards:
            value = CardCalculator.get_card_value(card)
            if value == BlackjackConfig.ACE_HIGH_VALUE:  # Ace
                aces += 1
            total += value
        
        # Adjust for Aces
        while total > BlackjackConfig.BLACKJACK_VALUE and aces > 0:
            total -= 10  # Convert Ace from 11 to 1
            aces -= 1
        
        return total
    
    @staticmethod
    def is_blackjack(cards: List[str]) -> bool:
        """
        Check if hand is a blackjack (21 with exactly 2 cards)
        
        Args:
            cards: List of card strings
            
        Returns:
            True if hand is blackjack, False otherwise
        """
        return (len(cards) == 2 and 
                CardCalculator.calculate_hand_total(cards) == BlackjackConfig.BLACKJACK_VALUE)
    
    @staticmethod
    def is_soft_hand(cards: List[str]) -> bool:
        """
        Check if hand is soft (contains usable Ace as 11)
        
        Args:
            cards: List of card strings
            
        Returns:
            True if hand is soft, False otherwise
        """
        total = 0
        has_ace = False
        
        for card in cards:
            value = CardCalculator.get_card_value(card)
            if value == BlackjackConfig.ACE_HIGH_VALUE:
                has_ace = True
            total += value
        
        return has_ace and total <= BlackjackConfig.BLACKJACK_VALUE

class GameResultCalculator:
    """Calculator for game results and outcomes"""
    
    @staticmethod
    def determine_winner(player_cards: List[str], dealer_cards: List[str]) -> str:
        """
        Determine the winner of a blackjack game
        
        Args:
            player_cards: Player's cards
            dealer_cards: Dealer's cards
            
        Returns:
            Result string: 'win', 'lose', 'push', 'blackjack', 'bust'
        """
        player_total = CardCalculator.calculate_hand_total(player_cards)
        dealer_total = CardCalculator.calculate_hand_total(dealer_cards)
        
        player_bj = CardCalculator.is_blackjack(player_cards)
        dealer_bj = CardCalculator.is_blackjack(dealer_cards)
        
        # Player busted
        if player_total > BlackjackConfig.BLACKJACK_VALUE:
            return 'bust'
        
        # Blackjack scenarios
        if player_bj and dealer_bj:
            return 'push'
        elif player_bj and not dealer_bj:
            return 'blackjack'
        elif not player_bj and dealer_bj:
            return 'lose'
        
        # Dealer busted
        if dealer_total > BlackjackConfig.BLACKJACK_VALUE:
            return 'win'
        
        # Normal comparison
        if player_total > dealer_total:
            return 'win'
        elif player_total < dealer_total:
            return 'lose'
        else:
            return 'push'

class FileManager:
    """Handle file operations for save/load functionality"""
    
    @staticmethod
    def generate_filename() -> str:
        """Generate a timestamp-based filename for data export"""
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{AppConfig.DATA_FILE_PREFIX}{timestamp}{AppConfig.DATA_FILE_EXTENSION}"
    
    @staticmethod
    def validate_json_structure(data: dict) -> bool:
        """
        Validate that loaded JSON has the expected structure
        
        Args:
            data: Dictionary loaded from JSON
            
        Returns:
            True if structure is valid, False otherwise
        """
        required_keys = ['games', 'stats']
        return all(key in data for key in required_keys)

class StatisticsCalculator:
    """Calculate various statistics for game analysis"""
    
    @staticmethod
    def calculate_win_rate(wins: int, total_games: int) -> float:
        """Calculate win percentage"""
        if total_games == 0:
            return 0.0
        return (wins / total_games) * 100
    
    @staticmethod
    def calculate_percentages(wins: int, losses: int, pushes: int) -> Tuple[float, float, float]:
        """Calculate win, loss, and push percentages"""
        total = wins + losses + pushes
        if total == 0:
            return (0.0, 0.0, 0.0)
        
        return (
            round((wins / total) * 100, 2),
            round((losses / total) * 100, 2),
            round((pushes / total) * 100, 2)
        )