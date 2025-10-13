import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random

def card_value(card_str):
    card_str = card_str.upper().strip()
    if card_str in ['J', 'Q', 'K', '10']:
        return 10
    elif card_str == 'A':
        return 11
    elif card_str.isdigit() and 2 <= int(card_str) <= 9:
        return int(card_str)
    else:
        raise ValueError(f"Geçersiz kart: {card_str}. Geçerli: 2-10, J, Q, K, A")

def random_card():
    ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
    return random.choice(ranks)

def random_card_str():
    return random.choice(card_options)

class Shoe:
    def __init__(self, decks=6):
        ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
        self.cards = ranks * 4 * max(1, int(decks))
        random.shuffle(self.cards)

    def draw(self):
        if not self.cards:
            # If shoe is empty, replenish with a new shoe
            self.__init__(decks=6)
        return self.cards.pop()

class BlackjackBot:
    def __init__(self):
        self.total_profit = 0
        self.wins = 0
        self.losses = 0
        self.pushes = 0
        self.hands_played = 0

    def update_stats(self, profit):
        self.total_profit += profit
        self.hands_played += 1
        if profit > 0:
            self.wins += 1
        elif profit < 0:
            self.losses += 1
        else:
            self.pushes += 1

    def get_ev(self):
        if self.hands_played == 0:
            return 0
        return self.total_profit / self.hands_played

    def get_percentages(self):
        total = self.wins + self.losses + self.pushes
        if total == 0:
            return (0, 0, 0)
        return (
            round(self.wins / total * 100, 2),
            round(self.losses / total * 100, 2),
            round(self.pushes / total * 100, 2)
        )

    def hand_total(self, cards):
        """Calculate hand total, handling aces properly"""
        total = sum(cards)
        aces = cards.count(11)
        
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        
        return total
    
    def is_soft(self, cards):
        """Check if hand is soft (contains usable ace)"""
        total = sum(cards)
        aces = cards.count(11)
        
        if aces == 0:
            return False
        
        # If we can use an ace as 11 without busting, it's soft
        return total <= 21
    
    def decide_action(self, player_cards, dealer_up):
        """Basic strategy decision making"""
        player_total = self.hand_total(player_cards)
        is_soft_hand = self.is_soft(player_cards)
        
        # Pair splitting logic
        if len(player_cards) == 2 and player_cards[0] == player_cards[1]:
            pair_value = player_cards[0]
            if pair_value == 11:  # Aces
                return 'split'
            elif pair_value == 8:  # 8s
                return 'split'
            elif pair_value == 9 and dealer_up not in [7, 10, 11]:
                return 'split'
            elif pair_value == 7 and dealer_up <= 7:
                return 'split'
            elif pair_value == 6 and dealer_up <= 6:
                return 'split'
            elif pair_value in [2, 3] and dealer_up <= 7:
                return 'split'
        
        # Soft hand strategy
        if is_soft_hand:
            if player_total >= 19:
                return 'stand'
            elif player_total == 18:
                if dealer_up in [2, 7, 8]:
                    return 'stand'
                elif dealer_up in [3, 4, 5, 6]:
                    return 'double'
                else:
                    return 'hit'
            elif player_total == 17:
                if dealer_up in [3, 4, 5, 6]:
                    return 'double'
                else:
                    return 'hit'
            elif player_total in [15, 16]:
                if dealer_up in [4, 5, 6]:
                    return 'double'
                else:
                    return 'hit'
            elif player_total in [13, 14]:
                if dealer_up in [5, 6]:
                    return 'double'
                else:
                    return 'hit'
            else:
                return 'hit'
        
        # Hard hand strategy
        if player_total >= 17:
            return 'stand'
        elif player_total >= 13:
            if dealer_up <= 6:
                return 'stand'
            else:
                return 'hit'
        elif player_total == 12:
            if dealer_up in [4, 5, 6]:
                return 'stand'
            else:
                return 'hit'
        elif player_total == 11:
            return 'double'
        elif player_total == 10:
            if dealer_up <= 9:
                return 'double'
            else:
                return 'hit'
        elif player_total == 9:
            if dealer_up in [3, 4, 5, 6]:
                return 'double'
            else:
                return 'hit'
        else:
            return 'hit'
    
    def play_player_hand(self, player_cards, dealer_up, shoe):
        """Play out player hand according to basic strategy"""
        hands = [player_cards.copy()]
        profits = []
        
        for hand in hands:
            bet = 1
            
            while True:
                action = self.decide_action(hand, dealer_up)
                
                if action == 'stand':
                    break
                elif action == 'hit':
                    hand.append(shoe.draw())
                    if self.hand_total(hand) > 21:
                        break
                elif action == 'double':
                    hand.append(shoe.draw())
                    bet *= 2
                    break
                elif action == 'split' and len(hand) == 2:
                    # Simple split handling - just play first hand
                    hand.append(shoe.draw())
                    break
            
            # Calculate profit for this hand
            player_total = self.hand_total(hand)
            if player_total > 21:
                profit = -bet
            else:
                dealer_total = self.play_dealer_hand(dealer_up, shoe)
                profit = self.calculate_profit(player_total, dealer_total, bet)
            
            profits.append(profit)
        
        return profits
    
    def play_dealer_hand(self, dealer_up, shoe):
        """Play dealer hand according to rules"""
        dealer_cards = [dealer_up, shoe.draw()]
        
        while self.hand_total(dealer_cards) < 17:
            dealer_cards.append(shoe.draw())
        
        return self.hand_total(dealer_cards)
    
    def calculate_profit(self, player_total, dealer_total, bet):
        """Calculate profit from a completed hand"""
        if player_total > 21:
            return -bet
        elif dealer_total > 21:
            return bet
        elif player_total > dealer_total:
            return bet
        elif player_total < dealer_total:
            return -bet
        else:
            return 0

    def simulate_specific_hand(self, player_cards, dealer_up_card, n=10000, decks=6):
        self.total_profit = 0
        self.wins = 0
        self.losses = 0
        self.pushes = 0
        self.hands_played = 0
        
        player = [card_value(c) for c in player_cards]
        dealer_up = card_value(dealer_up_card)
        
        if len(player) != 2:
            raise ValueError("Oyuncu elinde tam 2 kart olmalı.")

        # Handle blackjack case
        if self.hand_total(player) == 21:
            for _ in range(n):
                shoe = Shoe(decks=decks)
                # Remove the known cards from the shoe
                for card in player + [dealer_up]:
                    if card in shoe.cards:
                        shoe.cards.remove(card)
                
                dealer_hole = shoe.draw()
                if self.hand_total([dealer_up, dealer_hole]) == 21:
                    profit = 0
                else:
                    profit = 1.5
                self.update_stats(profit)
            return self.get_ev(), self.get_percentages()

        # Regular hand simulation
        for _ in range(n):
            shoe = Shoe(decks=decks)
            # Remove the known cards from the shoe
            for card in player + [dealer_up]:
                if card in shoe.cards:
                    shoe.cards.remove(card)
            
            dealer_hole = shoe.draw()
            profits = self.play_player_hand(player, dealer_up, shoe)
            self.update_stats(profits[0])

        return self.get_ev(), self.get_percentages()

def update_chart(win, loss, push):
    global chart_canvas, ax

    # Clear previous chart
    ax.clear()

    # Only show chart if there's data
    if win + loss + push > 0:
        labels = 'Kazanma', 'Kaybetme', 'Beraberlik'
        sizes = [win, loss, push]
        colors = ['#4CAF50', '#F44336', '#FFC107']

        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.axis('equal')
        ax.set_title("Sonuç Dağılımı")

    chart_canvas.draw()

def run_simulation():
    bot = BlackjackBot()
    try:
        if var_random.get():
            player_card1 = random_card_str()
            player_card2 = random_card_str()
            dealer_up = random_card_str()
            
            # Update the comboboxes to show the random cards
            combo_card1.set(player_card1)
            combo_card2.set(player_card2)
            combo_dealer.set(dealer_up)
        else:
            player_card1 = combo_card1.get().strip()
            player_card2 = combo_card2.get().strip()
            dealer_up = combo_dealer.get().strip()
            if not player_card1 or not player_card2 or not dealer_up:
                raise ValueError("Tüm kartları seçin veya rastgele seçin.")

        sims = int(entry_sims.get())
        decks = int(entry_decks.get())

        player_cards = [player_card1, player_card2]
        ev, percentages = bot.simulate_specific_hand(player_cards, dealer_up, n=sims, decks=decks)
        win, loss, push = percentages
        result_label.config(text=f"Beklenen Değer (EV): {ev:.4f} | Kazanma: {win}% | Kaybetme: {loss}% | Beraberlik: {push}%")
        update_chart(win, loss, push)
    except ValueError as e:
        messagebox.showerror("Hata", str(e))

card_options = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# Create main window
root = tk.Tk()
root.title("Blackjack Simülasyonu")
root.geometry("800x400")
root.resizable(False, False)  # Prevent window resizing

# Configure main grid
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)

# Create main frames with fixed sizes
left_frame = tk.Frame(root, width=400, height=400, bg='lightgray')
left_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
left_frame.grid_propagate(False)  # Prevent frame from shrinking
left_frame.grid_columnconfigure(0, weight=1)

right_frame = tk.Frame(root, width=350, height=400, bg='white')
right_frame.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
right_frame.grid_propagate(False)  # Prevent frame from shrinking

# Left panel - Controls with grid layout
tk.Label(left_frame, text="Oyuncu Kart 1:").grid(row=0, column=0, sticky='w', pady=2, padx=5)
combo_card1 = ttk.Combobox(left_frame, values=card_options, state="readonly", width=20)
combo_card1.grid(row=1, column=0, pady=2, padx=5, sticky='ew')

tk.Label(left_frame, text="Oyuncu Kart 2:").grid(row=2, column=0, sticky='w', pady=2, padx=5)
combo_card2 = ttk.Combobox(left_frame, values=card_options, state="readonly", width=20)
combo_card2.grid(row=3, column=0, pady=2, padx=5, sticky='ew')

tk.Label(left_frame, text="Dealer Upcard:").grid(row=4, column=0, sticky='w', pady=2, padx=5)
combo_dealer = ttk.Combobox(left_frame, values=card_options, state="readonly", width=20)
combo_dealer.grid(row=5, column=0, pady=2, padx=5, sticky='ew')

var_random = tk.BooleanVar()
check_random = tk.Checkbutton(left_frame, text="Rastgele El", variable=var_random)
check_random.grid(row=6, column=0, sticky='w', pady=5, padx=5)

tk.Label(left_frame, text="Simülasyon Sayısı:").grid(row=7, column=0, sticky='w', pady=2, padx=5)
entry_sims = ttk.Entry(left_frame, width=22)
entry_sims.insert(0, "10000")
entry_sims.grid(row=8, column=0, pady=2, padx=5, sticky='ew')

tk.Label(left_frame, text="Destek Sayısı (decks):").grid(row=9, column=0, sticky='w', pady=2, padx=5)
entry_decks = ttk.Entry(left_frame, width=22)
entry_decks.insert(0, "6")
entry_decks.grid(row=10, column=0, pady=2, padx=5, sticky='ew')

simulate_button = tk.Button(left_frame, text="Simüle Et", command=run_simulation, width=20)
simulate_button.grid(row=11, column=0, pady=10, padx=5)

# Result label with fixed height
result_frame = tk.Frame(left_frame, height=60)
result_frame.grid(row=12, column=0, sticky='ew', pady=10, padx=5)
result_frame.grid_propagate(False)  # Prevent resizing
result_frame.grid_columnconfigure(0, weight=1)

result_label = tk.Label(result_frame, text="Sonuç burada görünecek.", wraplength=350, justify='center')
result_label.grid(row=0, column=0, sticky='ew')

# Right panel - Chart with fixed positioning
right_frame.grid_columnconfigure(0, weight=1)
right_frame.grid_rowconfigure(0, weight=1)

fig = Figure(figsize=(4, 4))
ax = fig.add_subplot(111)
chart_canvas = FigureCanvasTkAgg(fig, master=right_frame)
chart_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

root.mainloop()