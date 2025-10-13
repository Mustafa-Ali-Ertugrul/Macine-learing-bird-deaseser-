#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Any
from utils import CardValidator, CardCalculator, GameResultCalculator, ValidationError
from config import AppConfig

class BlackjackAnalyzer:
    def __init__(self):
        self.games: List[Dict[str, Any]] = []
        self.stats = {
            'total_games': 0,
            'wins': 0,
            'losses': 0,
            'pushes': 0,
            'blackjacks': 0,
            'busts': 0
        }
    
    def add_game(self, player_cards: str, dealer_cards: str) -> None:
        """
        Add game result - automatically calculates outcome
        
        Args:
            player_cards: Player cards (e.g. "A,K" or "10,7,4")
            dealer_cards: Dealer cards (e.g. "Q,6,5")
            
        Raises:
            ValidationError: If card inputs are invalid
        """
        try:
            # Validate inputs
            player_card_list = CardValidator.validate_cards_input(player_cards)
            dealer_card_list = CardValidator.validate_cards_input(dealer_cards)
            
            # Calculate totals
            player_total = CardCalculator.calculate_hand_total(player_card_list)
            dealer_total = CardCalculator.calculate_hand_total(dealer_card_list)
            
            # Determine result
            result = GameResultCalculator.determine_winner(player_card_list, dealer_card_list)
            
            game_data = {
                'player_cards': player_cards,
                'dealer_cards': dealer_cards,
                'result': result,
                'player_total': player_total,
                'dealer_total': dealer_total
            }
            
            self.games.append(game_data)
            self.update_stats(result)
            
            print(f"Game #{len(self.games)} added:")
            print(f"  Player: {player_cards} (Total: {player_total})")
            print(f"  Dealer: {dealer_cards} (Total: {dealer_total})")
            print(f"  🎯 RESULT: {self.get_result_text(result)}")
            print()
            
        except ValidationError as e:
            raise ValidationError(f"Invalid game input: {str(e)}")
    

    
    def get_result_text(self, result):
        """Sonuç metnini döndür"""
        result_texts = {
            'win': '🎉 KAZANDIN!',
            'lose': '😞 KAYBETTİN',
            'push': '🤝 BERABERE',
            'blackjack': '🔥 BLACKJACK KAZANDIN!',
            'bust': '💥 PATLADIN!'
        }
        return result_texts.get(result, result)
    
    def update_stats(self, result):
        """İstatistikleri güncelle"""
        self.stats['total_games'] += 1
        
        if result == 'win':
            self.stats['wins'] += 1
        elif result == 'lose':
            self.stats['losses'] += 1
        elif result == 'push':
            self.stats['pushes'] += 1
        elif result == 'blackjack':
            self.stats['wins'] += 1
            self.stats['blackjacks'] += 1
        elif result == 'bust':
            self.stats['losses'] += 1
            self.stats['busts'] += 1
    
    def show_stats(self):
        """İstatistikleri göster"""
        if self.stats['total_games'] == 0:
            print("Henüz oyun verisi yok!")
            return
        
        win_rate = (self.stats['wins'] / self.stats['total_games']) * 100
        blackjack_rate = (self.stats['blackjacks'] / self.stats['total_games']) * 100
        bust_rate = (self.stats['busts'] / self.stats['total_games']) * 100
        push_rate = (self.stats['pushes'] / self.stats['total_games']) * 100
        
        print("=" * 50)
        print("BLACKJACK ANALİZ SONUÇLARI")
        print("=" * 50)
        print(f"Toplam Oyun: {self.stats['total_games']}")
        print(f"Kazanılan: {self.stats['wins']}")
        print(f"Kaybedilen: {self.stats['losses']}")
        print(f"Berabere: {self.stats['pushes']}")
        print(f"Blackjack: {self.stats['blackjacks']}")
        print(f"Bust: {self.stats['busts']}")
        print("-" * 30)
        print(f"Kazanma Oranı: {win_rate:.2f}%")
        print(f"Blackjack Oranı: {blackjack_rate:.2f}%")
        print(f"Bust Oranı: {bust_rate:.2f}%")
        print(f"Beraberlik Oranı: {push_rate:.2f}%")
        print("=" * 50)
    
    def show_game_history(self):
        """Oyun geçmişini göster"""
        if not self.games:
            print("Henüz oyun verisi yok!")
            return
        
        print("\nOYUN GEÇMİŞİ:")
        print("-" * 60)
        print(f"{'#':<3} {'Oyuncu':<20} {'Dealer':<20} {'Sonuç':<10}")
        print("-" * 60)
        
        for i, game in enumerate(self.games, 1):
            print(f"{i:<3} {game['player_cards']:<20} {game['dealer_cards']:<20} {game['result']:<10}")
    

    
    def interactive_mode(self):
        """İnteraktif mod"""
        print("🃏 BLACKJACK KAZANMA YÜZDESI ANALİZCİSİ 🃏")
        print("=" * 50)
        print("Komutlar:")
        print("  add - Yeni oyun ekle")
        print("  stats - İstatistikleri göster")
        print("  history - Oyun geçmişi")
        print("  clear - Tüm verileri temizle")
        print("  quit - Çıkış")
        print("=" * 50)
        
        while True:
            try:
                command = input("\nKomut girin: ").strip().lower()
                
                if command == 'add':
                    self.add_game_interactive()
                elif command == 'stats':
                    self.show_stats()
                elif command == 'history':
                    self.show_game_history()
                elif command == 'clear':
                    self.clear_data()
                elif command == 'quit':
                    print("Çıkılıyor...")
                    break
                else:
                    print("Geçersiz komut! (add/stats/history/clear/quit)")
            
            except KeyboardInterrupt:
                print("\nÇıkılıyor...")
                break
            except Exception as e:
                print(f"Hata: {e}")
    
    def add_game_interactive(self):
        """İnteraktif oyun ekleme"""
        try:
            print("\n--- YENİ OYUN EKLEME ---")
            player_cards = input("Oyuncu kartları (örn: A,K veya 10,7,4): ").strip()
            dealer_cards = input("Dealer kartları (örn: Q,6,5): ").strip()
            
            # Sonucu otomatik hesapla
            self.add_game(player_cards, dealer_cards)
            
        except Exception as e:
            print(f"Hata: {e}")
    
    def get_statistics(self):
        """İstatistikleri döndür"""
        return self.stats.copy()
    
    def clear_data(self):
        """Tüm verileri temizle"""
        confirm = input("Tüm verileri silmek istediğinizden emin misiniz? (evet/hayır): ")
        if confirm.lower() in ['evet', 'e', 'yes', 'y']:
            self.games = []
            self.stats = {
                'total_games': 0,
                'wins': 0,
                'losses': 0,
                'pushes': 0,
                'blackjacks': 0,
                'busts': 0
            }
            print("Tüm veriler temizlendi!")
        else:
            print("İptal edildi.")

# Hızlı test için örnek veriler
def quick_test():
    analyzer = BlackjackAnalyzer()
    
    # Örnek oyunlar
    analyzer.add_game("A,K", "Q,6,5")
    analyzer.add_game("10,7", "K,8")
    analyzer.add_game("9,9", "7,10,4")
    analyzer.add_game("K,6,5", "A,10")
    analyzer.add_game("A,8", "9,9")
    
    analyzer.show_stats()
    analyzer.show_game_history()

if __name__ == "__main__":
    # Hızlı test için uncomment et
    # quick_test()
    
    # İnteraktif mod
    analyzer = BlackjackAnalyzer()
    analyzer.interactive_mode()