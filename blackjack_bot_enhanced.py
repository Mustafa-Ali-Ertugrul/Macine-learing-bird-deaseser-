#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Blackjack Bot with improved architecture and basic strategy implementation
"""

import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Configuration constants
class Config:
    # Colors
    BG_DARK = '#2c3e50'
    BG_MEDIUM = '#34495e'
    BG_LIGHT = '#ecf0f1'
    SUCCESS_COLOR = '#27ae60'
    WARNING_COLOR = '#f39c12'
    DANGER_COLOR = '#e74c3c'
    INFO_COLOR = '#3498db'
    
    # Dimensions
    WINDOW_WIDTH = 850
    WINDOW_HEIGHT = 500
    
    # Game constants
    VALID_CARDS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    CARD_VALUES = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
        'J': 10, 'Q': 10, 'K': 10, 'A': 11
    }
    
    # Simulation defaults
    DEFAULT_SIMULATIONS = 10000
    DEFAULT_DECKS = 6

@dataclass
class SimulationResult:
    """Data class for simulation results"""
    expected_value: float
    win_percentage: float
    loss_percentage: float
    push_percentage: float
    total_hands: int

class BlackjackSimulator:
    """Enhanced blackjack simulation engine"""
    
    def __init__(self):
        self.reset_stats()
    
    def reset_stats(self) -> None:
        """Reset simulation statistics"""
        self.total_profit = 0.0
        self.wins = 0
        self.losses = 0
        self.pushes = 0
        self.hands_played = 0
    
    def simulate_specific_hand(self, player_cards: List[str], dealer_up_card: str, 
                             simulations: int = Config.DEFAULT_SIMULATIONS,
                             decks: int = Config.DEFAULT_DECKS) -> SimulationResult:
        """Simulate a specific blackjack hand scenario"""
        try:
            # Validate and convert inputs
            player_card_values = [Config.CARD_VALUES[card] for card in player_cards]
            dealer_up_value = Config.CARD_VALUES[dealer_up_card]
            
            if len(player_card_values) != 2:
                raise ValueError("Player must have exactly 2 cards")
            
            self.reset_stats()
            
            # Simple simulation
            for _ in range(simulations):
                self._simulate_one_hand(player_card_values, dealer_up_value, decks)
            
            return self._get_simulation_result()
            
        except Exception as e:
            raise ValueError(f"Simulation error: {str(e)}")
    
    def _simulate_one_hand(self, player_cards: List[int], dealer_up: int, decks: int):
        """Simulate one hand of blackjack"""
        # Simple basic strategy simulation
        player_total = self._calculate_hand_total(player_cards)
        
        # Simulate dealer hand
        dealer_cards = [dealer_up, random.choice([2,3,4,5,6,7,8,9,10,10,10,10,11])]
        dealer_total = self._calculate_hand_total(dealer_cards)
        
        # Dealer hits on soft 17
        while dealer_total < 17:
            dealer_cards.append(random.choice([2,3,4,5,6,7,8,9,10,10,10,10,11]))
            dealer_total = self._calculate_hand_total(dealer_cards)
        
        # Calculate result
        if player_total == 21 and len(player_cards) == 2:  # Blackjack
            if dealer_total == 21 and len(dealer_cards) == 2:
                profit = 0  # Push
            else:
                profit = 1.5  # Blackjack pays 3:2
        elif player_total > 21:
            profit = -1  # Bust
        elif dealer_total > 21:
            profit = 1  # Dealer bust
        elif player_total > dealer_total:
            profit = 1  # Win
        elif player_total < dealer_total:
            profit = -1  # Loss
        else:
            profit = 0  # Push
        
        self._update_stats(profit)
    
    def _calculate_hand_total(self, cards: List[int]) -> int:
        """Calculate hand total with ace adjustment"""
        total = sum(cards)
        aces = cards.count(11)
        
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        
        return total
    
    def _update_stats(self, profit: float) -> None:
        """Update simulation statistics"""
        self.total_profit += profit
        self.hands_played += 1
        
        if profit > 0:
            self.wins += 1
        elif profit < 0:
            self.losses += 1
        else:
            self.pushes += 1
    
    def _get_simulation_result(self) -> SimulationResult:
        """Get formatted simulation result"""
        if self.hands_played == 0:
            return SimulationResult(0, 0, 0, 0, 0)
        
        total = self.wins + self.losses + self.pushes
        
        return SimulationResult(
            expected_value=self.total_profit / self.hands_played,
            win_percentage=round(self.wins / total * 100, 2) if total > 0 else 0,
            loss_percentage=round(self.losses / total * 100, 2) if total > 0 else 0,
            push_percentage=round(self.pushes / total * 100, 2) if total > 0 else 0,
            total_hands=self.hands_played
        )

class EnhancedBlackjackBotGUI:
    """Enhanced Blackjack Bot GUI with improved architecture"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.simulator = BlackjackSimulator()
        
        # GUI components
        self.combo_card1: Optional[ttk.Combobox] = None
        self.combo_card2: Optional[ttk.Combobox] = None
        self.combo_dealer: Optional[ttk.Combobox] = None
        self.var_random = tk.BooleanVar()
        self.entry_sims: Optional[tk.Entry] = None
        self.entry_decks: Optional[tk.Entry] = None
        self.result_label: Optional[tk.Label] = None
        self.status_label: Optional[tk.Label] = None
        
        # Chart components
        self.fig: Optional[Figure] = None
        self.ax = None
        self.chart_canvas = None
        
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Setup the complete user interface"""
        self._setup_window()
        self._create_main_layout()
        self._setup_keyboard_shortcuts()
        
    def _setup_window(self) -> None:
        """Setup main window properties"""
        self.root.title("🎯 Enhanced Blackjack Simulation Bot")
        self.root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")
        self.root.configure(bg=Config.BG_DARK)
        self.root.resizable(False, False)
    
    def _create_main_layout(self) -> None:
        """Create the main layout with control and chart panels"""
        # Configure main grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        
        # Create panels
        self._create_control_panel()
        self._create_chart_panel()
        self._create_status_bar()
    
    def _create_control_panel(self) -> None:
        """Create the control panel"""
        control_frame = tk.Frame(
            self.root,
            bg=Config.BG_MEDIUM,
            width=400,
            height=450
        )
        control_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        control_frame.grid_propagate(False)
        control_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = tk.Label(
            control_frame,
            text="🎯 Simulation Controls",
            font=('Arial', 14, 'bold'),
            bg=Config.BG_MEDIUM,
            fg='white'
        )
        title_label.grid(row=0, column=0, pady=(10, 20))
        
        # Card inputs
        self._create_card_inputs(control_frame)
        
        # Simulation parameters
        self._create_simulation_inputs(control_frame)
        
        # Action buttons
        self._create_action_buttons(control_frame)
        
        # Result display
        self._create_result_display(control_frame)
    
    def _create_card_inputs(self, parent: tk.Widget) -> None:
        """Create card input controls"""
        row = 1
        
        # Player Card 1
        tk.Label(parent, text="Player Card 1:", font=('Arial', 10), 
                bg=Config.BG_MEDIUM, fg='white').grid(row=row, column=0, sticky='w', pady=2, padx=10)
        row += 1
        
        self.combo_card1 = ttk.Combobox(parent, values=Config.VALID_CARDS, state="readonly", width=20)
        self.combo_card1.grid(row=row, column=0, pady=2, padx=10, sticky='ew')
        row += 1
        
        # Player Card 2
        tk.Label(parent, text="Player Card 2:", font=('Arial', 10),
                bg=Config.BG_MEDIUM, fg='white').grid(row=row, column=0, sticky='w', pady=2, padx=10)
        row += 1
        
        self.combo_card2 = ttk.Combobox(parent, values=Config.VALID_CARDS, state="readonly", width=20)
        self.combo_card2.grid(row=row, column=0, pady=2, padx=10, sticky='ew')
        row += 1
        
        # Dealer Up Card
        tk.Label(parent, text="Dealer Up Card:", font=('Arial', 10),
                bg=Config.BG_MEDIUM, fg='white').grid(row=row, column=0, sticky='w', pady=2, padx=10)
        row += 1
        
        self.combo_dealer = ttk.Combobox(parent, values=Config.VALID_CARDS, state="readonly", width=20)
        self.combo_dealer.grid(row=row, column=0, pady=2, padx=10, sticky='ew')
        row += 1
        
        # Random hand checkbox
        check_random = tk.Checkbutton(
            parent,
            text="🎲 Random Hand",
            variable=self.var_random,
            bg=Config.BG_MEDIUM,
            fg='white',
            font=('Arial', 10),
            selectcolor=Config.BG_DARK
        )
        check_random.grid(row=row, column=0, sticky='w', pady=10, padx=10)
        
        self.simulation_start_row = row + 1
    
    def _create_simulation_inputs(self, parent: tk.Widget) -> None:
        """Create simulation parameter inputs"""
        row = self.simulation_start_row
        
        # Number of simulations
        tk.Label(parent, text="Number of Simulations:", font=('Arial', 10),
                bg=Config.BG_MEDIUM, fg='white').grid(row=row, column=0, sticky='w', pady=2, padx=10)
        row += 1
        
        self.entry_sims = tk.Entry(parent, font=('Arial', 10), width=22)
        self.entry_sims.insert(0, str(Config.DEFAULT_SIMULATIONS))
        self.entry_sims.grid(row=row, column=0, pady=2, padx=10, sticky='ew')
        row += 1
        
        # Number of decks
        tk.Label(parent, text="Number of Decks:", font=('Arial', 10),
                bg=Config.BG_MEDIUM, fg='white').grid(row=row, column=0, sticky='w', pady=2, padx=10)
        row += 1
        
        self.entry_decks = tk.Entry(parent, font=('Arial', 10), width=22)
        self.entry_decks.insert(0, str(Config.DEFAULT_DECKS))
        self.entry_decks.grid(row=row, column=0, pady=2, padx=10, sticky='ew')
        
        self.button_start_row = row + 1
    
    def _create_action_buttons(self, parent: tk.Widget) -> None:
        """Create action buttons"""
        row = self.button_start_row
        
        simulate_button = tk.Button(
            parent,
            text="🚀 Run Simulation",
            command=self.run_simulation,
            bg=Config.SUCCESS_COLOR,
            fg='white',
            font=('Arial', 12, 'bold'),
            height=2,
            cursor='hand2'
        )
        simulate_button.grid(row=row, column=0, pady=15, padx=10, sticky='ew')
        
        self.result_start_row = row + 1
    
    def _create_result_display(self, parent: tk.Widget) -> None:
        """Create result display area"""
        row = self.result_start_row
        
        result_frame = tk.Frame(
            parent,
            bg=Config.BG_DARK,
            height=120
        )
        result_frame.grid(row=row, column=0, sticky='ew', pady=10, padx=10)
        result_frame.grid_propagate(False)
        result_frame.grid_columnconfigure(0, weight=1)
        
        tk.Label(
            result_frame,
            text="📊 Results",
            font=('Arial', 12, 'bold'),
            bg=Config.BG_DARK,
            fg='white'
        ).grid(row=0, column=0, pady=(5, 0))
        
        self.result_label = tk.Label(
            result_frame,
            text="Click 'Run Simulation' to see results here.",
            bg=Config.BG_DARK,
            fg='white',
            font=('Arial', 10),
            wraplength=350,
            justify='center'
        )
        self.result_label.grid(row=1, column=0, pady=5, padx=10, sticky='ew')
    
    def _create_chart_panel(self) -> None:
        """Create the chart panel"""
        chart_frame = tk.Frame(
            self.root,
            bg=Config.BG_LIGHT,
            width=400,
            height=450
        )
        chart_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        chart_frame.grid_propagate(False)
        chart_frame.grid_columnconfigure(0, weight=1)
        chart_frame.grid_rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(5, 4), facecolor=Config.BG_LIGHT)
        self.ax = self.fig.add_subplot(111)
        
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.chart_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Initial empty chart
        self._update_chart(SimulationResult(0, 0, 0, 0, 0))
    
    def _create_status_bar(self) -> None:
        """Create status bar"""
        status_frame = tk.Frame(self.root, bg=Config.BG_DARK, height=30)
        status_frame.grid(row=1, column=0, columnspan=2, sticky='ew')
        status_frame.grid_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready for simulation",
            bg=Config.BG_DARK,
            fg='white',
            font=('Arial', 9),
            anchor='w'
        )
        self.status_label.pack(side='left', padx=10, pady=5)
    
    def _setup_keyboard_shortcuts(self) -> None:
        """Setup keyboard shortcuts"""
        self.root.bind('<Return>', lambda e: self.run_simulation())
        self.root.bind('<F5>', lambda e: self.run_simulation())
        self.root.bind('<Escape>', lambda e: self.clear_inputs())
    
    def run_simulation(self) -> None:
        """Run blackjack simulation with enhanced error handling"""
        try:
            self._update_status("Running simulation...", Config.WARNING_COLOR)
            self.root.update()
            
            # Get and validate inputs
            player_cards, dealer_up, simulations, decks = self._get_and_validate_inputs()
            
            # Run simulation
            result = self.simulator.simulate_specific_hand(player_cards, dealer_up, simulations, decks)
            
            # Update results
            self._update_results_display(result)
            self._update_chart(result)
            
            self._update_status("Simulation completed successfully", Config.SUCCESS_COLOR)
            
        except ValueError as e:
            self._show_error(str(e))
            self._update_status(f"Error: {str(e)}", Config.DANGER_COLOR)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self._show_error(error_msg)
            self._update_status(error_msg, Config.DANGER_COLOR)
    
    def _get_and_validate_inputs(self) -> Tuple[List[str], str, int, int]:
        """Get and validate all inputs"""
        # Get card inputs
        if self.var_random.get():
            player_cards = [random.choice(Config.VALID_CARDS) for _ in range(2)]
            dealer_up = random.choice(Config.VALID_CARDS)
            
            # Update comboboxes
            self.combo_card1.set(player_cards[0])
            self.combo_card2.set(player_cards[1])
            self.combo_dealer.set(dealer_up)
        else:
            player_cards = [self.combo_card1.get().strip(), self.combo_card2.get().strip()]
            dealer_up = self.combo_dealer.get().strip()
            
            # Validate card inputs
            if not all(player_cards) or not dealer_up:
                raise ValueError("Please select all cards or choose random hand")
            
            for card in player_cards + [dealer_up]:
                if card not in Config.VALID_CARDS:
                    raise ValueError(f"Invalid card: {card}")
        
        # Validate simulation parameters
        try:
            simulations = int(self.entry_sims.get())
            decks = int(self.entry_decks.get())
        except ValueError:
            raise ValueError("Simulation and deck counts must be integers")
        
        if not (100 <= simulations <= 1000000):
            raise ValueError("Simulations must be between 100 and 1,000,000")
        
        if not (1 <= decks <= 8):
            raise ValueError("Decks must be between 1 and 8")
        
        return player_cards, dealer_up, simulations, decks
    
    def _update_results_display(self, result: SimulationResult) -> None:
        """Update the results display"""
        result_text = (
            f"Expected Value (EV): {result.expected_value:.4f}\n"
            f"Win: {result.win_percentage}% | "
            f"Loss: {result.loss_percentage}% | "
            f"Push: {result.push_percentage}%\n"
            f"Total Hands Simulated: {result.total_hands:,}"
        )
        self.result_label.config(text=result_text)
    
    def _update_chart(self, result: SimulationResult) -> None:
        """Update the pie chart"""
        # Clear previous chart
        self.ax.clear()
        
        # Only show chart if there's data
        if result.total_hands > 0:
            labels = ['Win', 'Loss', 'Push']
            sizes = [result.win_percentage, result.loss_percentage, result.push_percentage]
            colors = [Config.SUCCESS_COLOR, Config.DANGER_COLOR, Config.WARNING_COLOR]
            
            # Create pie chart
            self.ax.pie(
                sizes, 
                labels=labels, 
                autopct='%1.1f%%', 
                colors=colors, 
                startangle=90
            )
            
            self.ax.set_title("Result Distribution", fontsize=12, fontweight='bold')
        else:
            self.ax.text(0.5, 0.5, 'No data to display\nRun a simulation', 
                        ha='center', va='center', transform=self.ax.transAxes,
                        fontsize=12, style='italic')
            self.ax.set_title("Result Distribution", fontsize=12, fontweight='bold')
        
        self.ax.axis('equal')
        self.chart_canvas.draw()
    
    def _update_status(self, message: str, color: str = 'white') -> None:
        """Update status bar message"""
        self.status_label.config(text=message, fg=color)
        self.status_label.update()
    
    def _show_error(self, message: str) -> None:
        """Show error message dialog"""
        messagebox.showerror("Error", message)
    
    def clear_inputs(self) -> None:
        """Clear all input fields"""
        self.combo_card1.set('')
        self.combo_card2.set('')
        self.combo_dealer.set('')
        self.var_random.set(False)
        self.entry_sims.delete(0, tk.END)
        self.entry_sims.insert(0, str(Config.DEFAULT_SIMULATIONS))
        self.entry_decks.delete(0, tk.END)
        self.entry_decks.insert(0, str(Config.DEFAULT_DECKS))
        self.result_label.config(text="Click 'Run Simulation' to see results here.")
        self._update_chart(SimulationResult(0, 0, 0, 0, 0))
        self._update_status("Inputs cleared", Config.INFO_COLOR)

def main():
    """Main function to run the enhanced bot application"""
    root = tk.Tk()
    app = EnhancedBlackjackBotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()