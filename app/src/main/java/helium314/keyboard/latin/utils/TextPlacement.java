package helium314.keyboard.latin.utils;

public final class TextPlacement {
    public String text;
    public final int selectionStart;

    public TextPlacement(String text, int selectionStart) {
        this.text = text;
        this.selectionStart = selectionStart;
    }

    public int selectionEnd() {
        return selectionStart + text.length();
    }
}
