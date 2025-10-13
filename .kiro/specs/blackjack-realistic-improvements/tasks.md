# Implementation Plan

- [ ] 1. Create core card and deck infrastructure
  - Implement Card dataclass with rank, suit, and value calculation
  - Create Deck class with 52-card generation and shuffle functionality
  - Write unit tests for card value calculations and deck operations
  - _Requirements: 1.1, 1.2, 1.6_

- [ ] 2. Implement DeckManager with realistic deck simulation
  - Create DeckManager class with multi-deck support (1-8 decks)
  - Implement card dealing with removal from available cards
  - Add automatic reshuffle when deck runs low (< 10 cards)
  - Write tests for deck depletion and reshuffle scenarios
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 3. Create enhanced Hand class with split support
  - Extend Hand class to support split hands and betting amounts
  - Implement can_split() method for pair detection
  - Add is_split_aces flag for special Ace splitting rules
  - Create hand completion status tracking
  - Write unit tests for hand operations and split detection
  - _Requirements: 2.1, 2.3, 2.4_

- [ ] 4. Implement HandManager for multiple hand coordination
  - Create HandManager class to coordinate multiple hands during splits
  - Implement split_hand() method to create two hands from one
  - Add current hand tracking and navigation between split hands
  - Handle bet doubling when splitting pairs
  - Write tests for split hand creation and management
  - _Requirements: 2.2, 2.5, 2.6_

- [ ] 5. Complete basic strategy tables
  - Add missing hard hand strategies for values 5-7 and 18-21
  - Implement complete soft hand strategy table (A,2 through A,9)
  - Create pair splitting strategy table with Y/N decisions for all pairs (A,A through 10,10)
  - Add split_strategy dictionary with optimal splitting decisions against all dealer cards
  - Write comprehensive tests for all strategy table lookups including split recommendations
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 6. Create StrategyEngine with advanced recommendations
  - Implement StrategyEngine class with complete strategy tables including split_strategy
  - Add get_pair_action() method that returns 'Y'/'N' for split recommendations
  - Implement get_strategy_action() method that checks for pairs first, then uses hard/soft tables
  - Add explain_action() method for strategy reasoning (e.g., "Always split Aces", "Never split 10s")
  - Handle strategy variations based on dealer rules
  - Write tests for strategy recommendations in all scenarios including pair splitting
  - _Requirements: 3.3, 3.5, 3.6_

- [ ] 7. Implement GameRules configuration system
  - Create GameRules class with dealer soft 17 options
  - Implement should_dealer_hit() method with H17/S17 logic
  - Add rule persistence with save/load functionality
  - Create rule description and validation methods
  - Write tests for different dealer rule configurations
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.6_

- [ ] 8. Update dealer logic for configurable rules
  - Modify play_dealer_hand() to use GameRules for soft 17 decisions
  - Implement proper soft hand detection for dealer
  - Update strategy recommendations based on dealer rules
  - Test dealer behavior with both H17 and S17 rules
  - _Requirements: 4.2, 4.3, 4.4_

- [ ] 9. Integrate split functionality into game flow
  - Modify play_hand() method to handle split scenarios
  - Implement split hand betting and balance validation
  - Add split hand evaluation against dealer
  - Handle special Ace splitting rules (one card only)
  - Write integration tests for complete split game scenarios
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.7_

- [ ] 10. Enhance GUI for split hand display
  - Update BlackjackGUI to show multiple hands during splits
  - Add visual indicators for current active hand
  - Implement split button and hand separation in interface
  - Display deck count and remaining cards information
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 11. Add deck information and rule display to GUI
  - Show remaining cards in deck counter
  - Display current dealer rules (H17/S17) in interface
  - Add deck reshuffle notifications
  - Implement rule configuration dialog
  - _Requirements: 5.1, 5.4, 5.5_

- [ ] 12. Implement enhanced statistics tracking
  - Extend StatisticsTracker to handle split hand statistics
  - Track performance by dealer rule configuration
  - Add split-specific statistics (split wins/losses)
  - Implement statistics export with configuration details
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 13. Update main BlackjackBot class integration
  - Integrate DeckManager into BlackjackBot constructor
  - Replace random card generation with deck-based dealing
  - Update game initialization to use GameRules
  - Modify statistics updates to handle split hands
  - _Requirements: 1.1, 1.2, 4.5_

- [ ] 14. Add comprehensive error handling and validation
  - Implement error handling for empty deck scenarios
  - Add validation for split operations and insufficient balance
  - Handle edge cases in strategy table lookups
  - Add input validation for rule configuration
  - Write tests for all error scenarios
  - _Requirements: 1.5, 2.7, 3.5_

- [ ] 15. Create integration tests and final validation
  - Write end-to-end tests for complete game scenarios with splits
  - Test deck management through multiple games
  - Validate strategy recommendations with new rule configurations
  - Performance test with large numbers of simulated games
  - _Requirements: All requirements validation_