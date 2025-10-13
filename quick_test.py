from blackjack_analyzer import BlackjackAnalyzer

# Hızlı test
analyzer = BlackjackAnalyzer()

# Örnek oyunlar ekle - sonuçlar otomatik hesaplanacak
print("Örnek oyunlar ekleniyor...")
analyzer.add_game("A,K", "Q,6,5", 100)      # Blackjack vs 21 -> Push
analyzer.add_game("10,7", "K,8", 50)        # 17 vs 18 -> Lose
analyzer.add_game("9,9", "7,10,4", 75)      # 18 vs 21 -> Lose
analyzer.add_game("K,6,5", "A,10", 100)     # 21 vs Blackjack -> Lose
analyzer.add_game("A,8", "9,9", 50)         # 19 vs 18 -> Win
analyzer.add_game("A,K", "7,6", 80)         # Blackjack vs 13 -> Blackjack!
analyzer.add_game("10,6,7", "K,Q", 60)      # 23 vs 20 -> Bust

# İstatistikleri göster
analyzer.show_stats()
analyzer.show_game_history()