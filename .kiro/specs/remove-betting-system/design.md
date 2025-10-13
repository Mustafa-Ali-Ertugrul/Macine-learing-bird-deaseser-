# Design Document

## Overview

The design focuses on systematically removing all betting-related functionality from the blackjack application while preserving the core game analysis capabilities. The application consists of three main components: the GUI (blackjack_gui.py), the analyzer (blackjack_analyzer.py), and the bot (blackjack_bot.py). Each component requires modifications to eliminate betting logic while maintaining clean, functional code.

## Architecture

The application maintains its existing three-layer architecture:

1. **Presentation Layer (blackjack_gui.py)**: Tkinter-based GUI for user interaction
2. **Business Logic Layer (blackjack_analyzer.py)**: Core game analysis and statistics
3. **Simulation Layer (blackjack_bot.py)**: Advanced blackjack bot with strategy implementation

The design preserves the separation of concerns while removing all monetary calculations and betting-related state management.

## Components and Interfaces

### BlackjackGUI Component

**Modified Interface:**
- Remove betting amount input field and related UI elements
- Simplify the input form to only include player cards and dealer cards
- Update