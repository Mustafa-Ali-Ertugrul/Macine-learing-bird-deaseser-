#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
from datetime import datetime
from blackjack_analyzer import BlackjackAnalyzer

class BlackjackGUI:
    def __init__(self, root):
        self.root = root
        self.analyzer = BlackjackAnalyzer()
        self.setup_ui()
        
    def setup_ui(self):
        self.root.title("🃏 Blackjack Kazanma Yüzdesi Analizi")
        self.root.geometry("800x600")
        self.root.configure(bg='#0f5132')
        
        # Ana stil
        style = ttk.Style()
        style.theme_use('clam')
        
        # Özel stiller
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#0f5132', foreground='white')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#0f5132', foreground='white')
        style.configure('Info.TLabel', font=('Arial', 10), background='#0f5132', foreground='white')
        style.configure('Success.TLabel', font=('Arial', 12, 'bold'), background='#0f5132', foreground='#28a745')
        style.configure('Danger.TLabel', font=('Arial', 12, 'bold'), background='#0f5132', foreground='#dc3545')
        
        # Ana başlık
        title_frame = tk.Frame(self.root, bg='#0f5132')
        title_frame.pack(fill='x', padx=10, pady=10)
        
        title_label = ttk.Label(title_frame, text="🃏 BLACKJACK KAZANMA YÜZDESİ ANALİZİ", style='Title.TLabel')
        title_label.pack()
        
        # Ana container
        main_frame = tk.Frame(self.root, bg='#0f5132')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Configure grid weights for main_frame
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Sol panel - Oyun ekleme
        left_frame = tk.LabelFrame(main_frame, text="YENİ OYUN EKLE", bg='#198754', fg='white', font=('Arial', 12, 'bold'), width=380, height=300)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5), pady=0)
        left_frame.grid_propagate(False)  # Prevent resizing
        
        self.create_input_section(left_frame)
        
        # Sağ panel - İstatistikler
        right_frame = tk.LabelFrame(main_frame, text="İSTATİSTİKLER", bg='#198754', fg='white', font=('Arial', 12, 'bold'), width=380, height=300)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0), pady=0)
        right_frame.grid_propagate(False)  # Prevent resizing
        
        self.create_stats_section(right_frame)
        
        # Alt panel - Oyun geçmişi
        bottom_frame = tk.LabelFrame(self.root, text="OYUN GEÇMİŞİ", bg='#198754', fg='white', font=('Arial', 12, 'bold'))
        bottom_frame.pack(fill='both', expand=True, padx=10, pady=(5, 10))
        
        self.create_history_section(bottom_frame)
        
    def create_input_section(self, parent):
        # Configure parent grid
        parent.grid_columnconfigure(0, weight=1)
        
        # Oyuncu kartları
        tk.Label(parent, text="Oyuncu Kartları:", bg='#198754', fg='white', font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', padx=10, pady=(10, 5))
        self.player_entry = tk.Entry(parent, font=('Arial', 12), width=25)
        self.player_entry.grid(row=1, column=0, padx=10, pady=(0, 5), sticky='ew')
        tk.Label(parent, text="Örnek: A,K veya 10,7,4", bg='#198754', fg='#ffc107', font=('Arial', 8)).grid(row=2, column=0, padx=10, pady=(0, 10), sticky='w')
        
        # Dealer kartları
        tk.Label(parent, text="Dealer Kartları:", bg='#198754', fg='white', font=('Arial', 10, 'bold')).grid(row=3, column=0, sticky='w', padx=10, pady=(0, 5))
        self.dealer_entry = tk.Entry(parent, font=('Arial', 12), width=25)
        self.dealer_entry.grid(row=4, column=0, padx=10, pady=(0, 5), sticky='ew')
        tk.Label(parent, text="Örnek: Q,6,5", bg='#198754', fg='#ffc107', font=('Arial', 8)).grid(row=5, column=0, padx=10, pady=(0, 10), sticky='w')
        

        
        # Butonlar
        button_frame = tk.Frame(parent, bg='#198754')
        button_frame.grid(row=6, column=0, sticky='ew', padx=10, pady=10)
        button_frame.grid_columnconfigure(0, weight=1)
        
        add_btn = tk.Button(button_frame, text="🎯 OYUN EKLE", command=self.add_game, 
                           bg='#28a745', fg='white', font=('Arial', 12, 'bold'), 
                           relief='raised', bd=3, cursor='hand2', height=2)
        add_btn.grid(row=0, column=0, sticky='ew', pady=(0, 5))
        
        clear_btn = tk.Button(button_frame, text="🗑️ TEMİZLE", command=self.clear_inputs,
                             bg='#ffc107', fg='black', font=('Arial', 10, 'bold'),
                             relief='raised', bd=2, cursor='hand2', height=1)
        clear_btn.grid(row=1, column=0, sticky='ew', pady=(0, 5))
        
        # Sonuç gösterimi
        self.result_frame = tk.Frame(parent, bg='#198754', height=80)
        self.result_frame.grid(row=7, column=0, sticky='ew', padx=10, pady=10)
        self.result_frame.grid_propagate(False)  # Prevent resizing
        self.result_frame.grid_columnconfigure(0, weight=1)
        
        self.result_label = tk.Label(self.result_frame, text="", bg='#198754', fg='white', 
                                    font=('Arial', 12, 'bold'), wraplength=300, justify='center')
        self.result_label.grid(row=0, column=0, sticky='ew')
        
    def create_stats_section(self, parent):
        # Configure parent grid
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        
        # İstatistik labelları
        self.stats_frame = tk.Frame(parent, bg='#198754')
        self.stats_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        self.stats_frame.grid_columnconfigure(0, weight=1)
        
        # İstatistik değişkenleri
        self.total_games_var = tk.StringVar(value="0")
        self.win_rate_var = tk.StringVar(value="0.00%")
        self.wins_var = tk.StringVar(value="0")
        self.losses_var = tk.StringVar(value="0")
        
        # İstatistik gösterimi
        stats_data = [
            ("📊 Toplam Oyun:", self.total_games_var),
            ("🎯 Kazanma Oranı:", self.win_rate_var),
            ("✅ Kazanılan:", self.wins_var),
            ("❌ Kaybedilen:", self.losses_var)
        ]
        
        for i, (label, var) in enumerate(stats_data):
            frame = tk.Frame(self.stats_frame, bg='#198754', height=30)
            frame.grid(row=i, column=0, sticky='ew', pady=2)
            frame.grid_propagate(False)  # Prevent resizing
            frame.grid_columnconfigure(1, weight=1)
            
            tk.Label(frame, text=label, bg='#198754', fg='white', 
                    font=('Arial', 11, 'bold'), width=15, anchor='w').grid(row=0, column=0, sticky='w', padx=(0, 10))
            tk.Label(frame, textvariable=var, bg='#198754', fg='#ffc107', 
                    font=('Arial', 11, 'bold'), width=10, anchor='e').grid(row=0, column=1, sticky='e')
        
        # Detaylı istatistikler butonu
        detail_btn = tk.Button(self.stats_frame, text="📋 DETAYLI İSTATİSTİK", 
                              command=self.show_detailed_stats,
                              bg='#17a2b8', fg='white', font=('Arial', 10, 'bold'),
                              relief='raised', bd=2, cursor='hand2', height=2)
        detail_btn.grid(row=5, column=0, sticky='ew', pady=(15, 5))
        
        # Veri kaydetme butonu
        save_btn = tk.Button(self.stats_frame, text="💾 VERİLERİ KAYDET", 
                            command=self.save_data,
                            bg='#6f42c1', fg='white', font=('Arial', 10, 'bold'),
                            relief='raised', bd=2, cursor='hand2', height=2)
        save_btn.grid(row=6, column=0, sticky='ew', pady=5)
        
        # Veri yükleme butonu
        load_btn = tk.Button(self.stats_frame, text="📂 VERİLERİ YÜKLE", 
                            command=self.load_data,
                            bg='#fd7e14', fg='white', font=('Arial', 10, 'bold'),
                            relief='raised', bd=2, cursor='hand2', height=2)
        load_btn.grid(row=7, column=0, sticky='ew', pady=5)
        
    def create_history_section(self, parent):
        # Treeview için frame
        tree_frame = tk.Frame(parent, bg='#198754')
        tree_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Treeview
        columns = ('Oyun', 'Oyuncu', 'Dealer', 'Sonuç')
        self.history_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=8)
        
        # Sütun başlıkları
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=120, anchor='center')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack
        self.history_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Temizleme butonu
        clear_history_btn = tk.Button(parent, text="🗑️ GEÇMİŞİ TEMİZLE", 
                                     command=self.clear_history,
                                     bg='#dc3545', fg='white', font=('Arial', 10, 'bold'),
                                     relief='raised', bd=2, cursor='hand2')
        clear_history_btn.pack(pady=(5, 10))
        
    def add_game(self):
        try:
            player_cards = self.player_entry.get().strip()
            dealer_cards = self.dealer_entry.get().strip()
            
            if not player_cards or not dealer_cards:
                messagebox.showerror("Hata", "Lütfen tüm alanları doldurun!")
                return
            
            # Oyunu ekle
            self.analyzer.add_game(player_cards, dealer_cards)
            
            # Son oyunun sonucunu göster
            last_game = self.analyzer.games[-1]
            result_text = self.analyzer.get_result_text(last_game['result'])
            
            self.result_label.config(text=f"🎯 {result_text}\n"
                                         f"Oyuncu: {last_game['player_total']} | "
                                         f"Dealer: {last_game['dealer_total']}")
            
            # Arayüzü güncelle
            self.update_stats()
            self.update_history()
            self.clear_inputs()
            
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {str(e)}")
    
    def clear_inputs(self):
        self.player_entry.delete(0, tk.END)
        self.dealer_entry.delete(0, tk.END)
        
    def update_stats(self):
        stats = self.analyzer.get_statistics()
        
        self.total_games_var.set(str(stats['total_games']))
        self.wins_var.set(str(stats['wins']))
        self.losses_var.set(str(stats['losses']))
        
        if stats['total_games'] > 0:
            win_rate = (stats['wins'] / stats['total_games']) * 100
            self.win_rate_var.set(f"{win_rate:.2f}%")
        else:
            self.win_rate_var.set("0.00%")
    
    def update_history(self):
        # Mevcut verileri temizle
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # Yeni verileri ekle
        for i, game in enumerate(self.analyzer.games, 1):
            self.history_tree.insert('', 'end', values=(
                i,
                f"{game['player_cards']} ({game['player_total']})",
                f"{game['dealer_cards']} ({game['dealer_total']})",
                game['result'].upper()
            ))
    
    def show_detailed_stats(self):
        if self.analyzer.stats['total_games'] == 0:
            messagebox.showinfo("Bilgi", "Henüz oyun verisi yok!")
            return
        
        # Yeni pencere
        detail_window = tk.Toplevel(self.root)
        detail_window.title("Detaylı İstatistikler")
        detail_window.geometry("400x500")
        detail_window.configure(bg='#0f5132')
        
        # Scrolled text
        text_widget = scrolledtext.ScrolledText(detail_window, wrap=tk.WORD, 
                                               font=('Courier', 10), bg='#212529', fg='white')
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        
        # İstatistikleri hazırla
        stats = self.analyzer.get_statistics()
        
        detail_text = f"""
🃏 DETAYLI BLACKJACK İSTATİSTİKLERİ
{'='*50}

📊 GENEL BİLGİLER:
  • Toplam Oyun: {stats['total_games']}
  • Kazanılan: {stats['wins']}
  • Kaybedilen: {stats['losses']}
  • Berabere: {stats['pushes']}
  • Blackjack: {stats['blackjacks']}
  • Bust: {stats['busts']}

🎯 PERFORMANS ANALİZİ:
  • Kazanma Oranı: {(stats['wins']/max(stats['total_games'], 1)*100):.2f}%
  • Blackjack Oranı: {(stats['blackjacks']/max(stats['total_games'], 1)*100):.2f}%
  • Bust Oranı: {(stats['busts']/max(stats['total_games'], 1)*100):.2f}%
  • Beraberlik Oranı: {(stats['pushes']/max(stats['total_games'], 1)*100):.2f}%

📅 RAPOR TARİHİ: {datetime.now().strftime('%d.%m.%Y %H:%M')}
"""
        
        text_widget.insert(tk.END, detail_text)
        text_widget.config(state='disabled')
    
    def clear_history(self):
        if messagebox.askyesno("Onay", "Tüm oyun geçmişini silmek istediğinizden emin misiniz?"):
            self.analyzer.games = []
            self.analyzer.stats = {
                'total_games': 0, 'wins': 0, 'losses': 0, 'pushes': 0,
                'blackjacks': 0, 'busts': 0
            }
            self.update_stats()
            self.update_history()
            self.result_label.config(text="")
            messagebox.showinfo("Bilgi", "Tüm veriler temizlendi!")
    
    def save_data(self):
        try:
            filename = f"blackjack_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            data = {
                'games': self.analyzer.games,
                'stats': self.analyzer.stats,
                'export_date': datetime.now().isoformat()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("Başarılı", f"Veriler '{filename}' dosyasına kaydedildi!")
            
        except Exception as e:
            messagebox.showerror("Hata", f"Kaydetme hatası: {str(e)}")
    
    def load_data(self):
        try:
            from tkinter import filedialog
            filename = filedialog.askopenfilename(
                title="Veri Dosyası Seç",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.analyzer.games = data.get('games', [])
                self.analyzer.stats = data.get('stats', {})
                
                self.update_stats()
                self.update_history()
                
                messagebox.showinfo("Başarılı", f"Veriler '{filename}' dosyasından yüklendi!")
                
        except Exception as e:
            messagebox.showerror("Hata", f"Yükleme hatası: {str(e)}")

def main():
    root = tk.Tk()
    app = BlackjackGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()