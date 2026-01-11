import numpy as np
import json
import chess
import pygame

from tensorflow.keras.models import load_model # type: ignore
model = load_model("model\\chess_model_v1.1.keras")

with open("model\\int_to_move_v1.1.json", "r") as f:
    int_to_move = json.load(f)
    #type cast to int because json keys are strings
    int_to_move = {int(k): v for k, v in int_to_move.items()}   

def board_to_matrix(board: chess.Board):
    matrix = np.zeros((8, 8, 12))
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[row, col, piece_type + piece_color] = 1
    return matrix

def predict_next_move(board: chess.Board):
    board_matrix = board_to_matrix(board).reshape(1, 8, 8, 12)
    predictions = model.predict(board_matrix)[0]
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(predictions)[::-1] 

    for move_index in sorted_indices:
        move = int_to_move[move_index]
        if move in legal_moves_uci:
            return move
    return None

def main():
    board = chess.Board()
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess")

    #font that supports chess symbols
    font = pygame.font.SysFont("segoeuisymbol", SQ_SIZE - 10)
    
    running = True
    selected_square = None
    bot_last_move = None
    turn = int(PLAY_AS_WHITE) #0 for ai first, 1 for player first
    while running:
        if board.is_game_over():
            running = False
            outcome = board.outcome()
            if outcome.winner is True:
                print("White Wins!")
            elif outcome.winner is False:
                print("Black Wins!")
            else:
                print("Draw!")
            break
        
        if not turn:
            # AI's turn
            pygame.time.delay(800)
            move = predict_next_move(board)
            board.push_uci(move)
            bot_last_move = move
            turn = 1
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Get the mouse position and convert it to a board coordinate
                    x, y = pygame.mouse.get_pos()
                    col = x // SQ_SIZE
                    row = 7 - (y // SQ_SIZE)  # Flip row because Pygame y starts at top
                    square = chess.square(col, row)

                    if selected_square is None:
                        #first click: select a piece
                        if board.piece_at(square):
                            selected_square = square
                    else:
                        #second click: try to make a move
                        move = chess.Move(selected_square, square)
                        
                        #pawn promotion
                        if board.piece_at(selected_square).piece_type == chess.PAWN:
                            if (chess.square_rank(square) == 7 and board.turn == chess.WHITE) or \
                            (chess.square_rank(square) == 0 and board.turn == chess.BLACK):
                                move.promotion = chess.QUEEN

                        if move in board.legal_moves:
                            board.push(move)
                            turn = 0
                        
                        selected_square = None # Reset for next move

        #Draw board background
        colors = [pygame.Color("#eeeed2"), pygame.Color("#769656")] 
        for r in range(8):
            for c in range(8):
                color = colors[((r + c) % 2)]
                pygame.draw.rect(screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

        #Draw pieces
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                symbol = UNICODE_PIECES[piece.symbol()]

                text_color = (0, 0, 0) if piece.color == chess.BLACK else (50, 50, 50)
                text_surface = font.render(symbol, True, text_color)
                
                # Center the text in the square
                row = 7 - (i // 8)
                col = i % 8
                text_rect = text_surface.get_rect(center=(col*SQ_SIZE + SQ_SIZE//2, row*SQ_SIZE + SQ_SIZE//2))
                screen.blit(text_surface, text_rect)

            if selected_square is not None:
                s_col = chess.square_file(selected_square)
                s_row = 7 - chess.square_rank(selected_square)
                highlight_rect = pygame.Rect(s_col * SQ_SIZE, s_row * SQ_SIZE, SQ_SIZE, SQ_SIZE)
                pygame.draw.rect(screen, (255, 0, 0), highlight_rect, 5)

        #highlight bot's last move
        if bot_last_move and turn:
            from_square = chess.square_file(chess.Move.from_uci(bot_last_move).from_square)
            from_row = 7 - chess.square_rank(chess.Move.from_uci(bot_last_move).from_square)
            to_square = chess.square_file(chess.Move.from_uci(bot_last_move).to_square)
            to_row = 7 - chess.square_rank(chess.Move.from_uci(bot_last_move).to_square)

            highlight_from = pygame.Rect(from_square * SQ_SIZE, from_row * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            highlight_to = pygame.Rect(to_square * SQ_SIZE, to_row * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            pygame.draw.rect(screen, (0, 255, 255), highlight_from, 5)
            pygame.draw.rect(screen, (0, 255, 255), highlight_to, 5)

        pygame.display.flip()
    
    pygame.time.delay(10000)
    pygame.quit()


WIDTH, HEIGHT = 640, 640
SQ_SIZE = WIDTH // 8
PLAY_AS_WHITE = True  # Set to False to play as Black

UNICODE_PIECES = {
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙'
}

main()